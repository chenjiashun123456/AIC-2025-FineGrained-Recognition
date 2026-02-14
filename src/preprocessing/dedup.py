import os
import argparse
from typing import List, Tuple, Dict
# 重复检测/去重工具：
# - 先按类别分组，避免跨类误报，提高效率
# - 使用两种近似重复判据：感知哈希（pHash）的汉明距离与特征余弦相似度
# - 特征来自 ResNet50 的中层（layer3 之后），维度 1024，更关注纹理/结构相似
# - 输出 CSV，包含可能重复的图片对与相似度评分，便于人工复核与清洗

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import imagehash
# 进度条（tqdm）用于长循环的可视化反馈；若不可用则降级为无输出
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, total=None, desc=None, unit=None):
        return iterable


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: str) -> List[Tuple[str, str]]:
    """遍历训练根目录，返回 (图片路径, 类别名) 列表。
    仅收集常见图片扩展名，以防止非法文件造成读入错误。
    """
    items = []
    classes = []
    for name in sorted(os.listdir(root)):
        cls_path = os.path.join(root, name)
        if not os.path.isdir(cls_path):
            continue
        classes.append(name)
    for c in classes:
        cls_path = os.path.join(root, c)
        for dp, _, fns in os.walk(cls_path):
            for fn in fns:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    items.append((os.path.join(dp, fn), c))
    return items


def compute_phash(path: str) -> int:
    """计算图片的感知哈希（pHash），用于近似重复检测。
    返回值为 16 进制字符串转整数，便于后续进行按位异或计算汉明距离。
    """
    with Image.open(path) as img:
        img = img.convert('RGB')
    return int(str(imagehash.phash(img)), 16)


class MidFeat(nn.Module):
    """中层特征提取网络（基于 ResNet50）。
    - 截取到 layer3 的特征并做 GAP，得到 1024 维向量。
    - 中层特征对纹理/结构敏感，适合做近似重复判断。
    """
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐层前向，最后做自适应平均池化并展平为 [B, 1024]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg(x).flatten(1)  # [B, 1024]
        return x


def build_tf(size: int = 384):
    """评估/特征提取使用的图像变换：Resize + CenterCrop + 标准化。"""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度，数值范围 [-1, 1]，越大越相似。"""
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def dedup(train_dir: str, save_csv: str, phash_thresh: int = 3, cos_thresh: float = 0.99, img_size: int = 384):
    """
    扫描训练集内的近似重复图片对：
    - 仅在同一类别内两两比较，降低计算量并减少跨类误判
    - 先用 pHash 汉明距离（<= phash_thresh）快速筛重；否则再用特征余弦相似度（>= cos_thresh）
    - 阈值可按数据规模与误判容忍度调整（pHash 越小越严格；cos 越大越严格）
    - 结果保存为 CSV：class, path_a, path_b, type(phash|cosine), score
    """
    items = list_images(train_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = build_tf(img_size)
    model = MidFeat().to(device).eval()

    feats = []
    hashes = []
    paths = []
    labels = []

    # 提取 pHash 与中层特征
    for path, cls in tqdm(items, desc='提取特征与pHash', unit='图像'):
        try:
            h = compute_phash(path)
            with Image.open(path) as img:
                img = img.convert('RGB')
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(x).cpu().numpy()[0]
        except Exception:
            continue
        hashes.append(h)
        feats.append(f)
        paths.append(path)
        labels.append(cls)

    # 按类别分桶后做成对比较，显著减少复杂度
    rows = []
    by_class: Dict[str, List[int]] = {}
    for i, cls in enumerate(labels):
        by_class.setdefault(cls, []).append(i)

    # 按类别成对比较
    for cls, idxs in tqdm(by_class.items(), desc='类内成对比较', unit='类'):
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                # 先计算 pHash 的汉明距离（越小说明越像）
                hdist = bin(hashes[a] ^ hashes[b]).count('1')
                if hdist <= phash_thresh:
                    rows.append({
                        'class': cls,
                        'path_a': paths[a],
                        'path_b': paths[b],
                        'type': 'phash',
                        'score': hdist,
                    })
                    continue
                # 若 pHash 不足以判重，再看特征余弦相似度（越大越像）
                cs = cosine_sim(np.array(feats[a]), np.array(feats[b]))
                if cs >= cos_thresh:
                    rows.append({
                        'class': cls,
                        'path_a': paths[a],
                        'path_b': paths[b],
                        'type': 'cosine',
                        'score': cs,
                    })

    # 导出结果到 CSV，便于人工核查与清洗
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved dedup pairs to {save_csv} with {len(df)} pairs")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', type=str, required=True)
    ap.add_argument('--save_csv', type=str, default=r"d:\Aic\fg4060_proj\fg4060_outputs\dedup_pairs.csv")
    ap.add_argument('--phash_thresh', type=int, default=3)
    ap.add_argument('--cos_thresh', type=float, default=0.995)
    ap.add_argument('--img_size', type=int, default=384)
    args = ap.parse_args()
    dedup(args.train_dir, args.save_csv, args.phash_thresh, args.cos_thresh, args.img_size)
