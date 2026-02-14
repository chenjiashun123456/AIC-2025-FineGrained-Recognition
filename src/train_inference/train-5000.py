import os
# 顶部导入清理
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import get_model
import time
from tqdm import tqdm
from collections import deque
import numpy as np
import json, glob, os


# 提前定义：仅保留当前最好的 K 个检查点
def get_gpu_stats(device_index=0, device=None):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return used_gb, total_gb, util
    except Exception:
        used_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(device or torch.device('cuda:0')).total_memory / (1024 ** 3)
        return used_gb, total_gb, 0

def keep_top_k_checkpoints(checkpoint_dir, new_item, k=5):
    board_path = os.path.join(checkpoint_dir, 'best_ckpts.json')
    board = []
    if os.path.exists(board_path):
        try:
            with open(board_path, 'r') as f:
                board = json.load(f)
        except Exception:
            board = []
    board.append(new_item)
    board.sort(key=lambda x: x['score'], reverse=True)
    keep = board[:k]
    with open(board_path, 'w') as f:
        json.dump(keep, f, indent=2)
    keep_paths = {item['path'] for item in keep}
    for p in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')):
        if p not in keep_paths:
            try:
                os.remove(p)
            except Exception:
                pass

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25, accum_steps=8, start_epoch=0, scheduler=None, checkpoint_dir=None):
    model.to(device)
    model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True
    # 与 train.py 对齐默认检查点根目录
    checkpoint_dir_base = os.getenv("CHECKPOINT_DIR", "/root/swinbase/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_txt_path = os.path.join(checkpoint_dir, "train_log.txt")

    # 正确的 GradScaler 用法
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    # 修复：确保恢复后的优化器状态在同一设备
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # 训练前的续训计划提示与防守
    print(f"开始训练：从第 {start_epoch+1} 轮到第 {num_epochs} 轮（共 {max(0, num_epochs-start_epoch)} 轮），loader_len={len(train_loader)}")
    if start_epoch >= num_epochs:
        print(f"无需训练：start_epoch({start_epoch}) >= num_epochs({num_epochs})，直接返回当前模型。")
        return model

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        optimizer.zero_grad()

        # 每轮滚动窗口
        losses_win = deque(maxlen=100)
        acc_win = deque(maxlen=100)

        # 批次级进度条
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', mininterval=0.5)

        for i, (inputs, labels) in pbar:
            inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device)
            # 保持 FP16 混合精度（ConvNeXt V2-Large 训练更稳、显存更省）
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0) * accum_steps
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 进度条后缀：滚动均值 + GPU 监控
            losses_win.append(loss.item() * accum_steps)
            acc_win.append((predicted == labels).float().mean().item())
            used_gb, total_gb, util = get_gpu_stats(0, device=device)
            current_lr_inbatch = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f'{np.mean(losses_win):.4f}',
                             acc=f'{np.mean(acc_win)*100:.2f}%',
                             lr=f'{current_lr_inbatch:.3e}',
                             gpu=f'{util}%',
                             mem=f'{used_gb:.1f}/{total_gb:.1f}GB')

        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        epoch_time = time.time() - start_time
        # 终端与文本日志同步
        line = (f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, '
                f'Acc: {epoch_acc:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s')
        print(line)
        with open(log_txt_path, 'a') as f:
            f.write(line + '\n')
        # 先构造检查点字典，再保存
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'acc': epoch_acc,
            'num_classes': int(getattr(model, 'num_classes', len(getattr(train_loader.dataset, 'classes', [])) or 0))
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        ck_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, ck_path)
        # 仅保留当前最好的5个
        score = epoch_acc - 0.2 * epoch_loss
        keep_top_k_checkpoints(checkpoint_dir, {
            'epoch': epoch + 1,
            'path': ck_path,
            'loss': float(epoch_loss),
            'acc': float(epoch_acc),
            'score': float(score)
        }, k=5)
    return model

def main():
    # 新增：命令行参数解析，支持 --train_dir 与默认本地路径
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=None,
                        help='训练数据目录（不传则使用本地5000类路径）')
    args = parser.parse_args()
    # 固定为5000类训练
    selected_class_num = 5000

    # 路径适配：训练数据目录（Windows 本地）
    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = r'd:\Aic\最终的代码\train_5000'

    model_name = 'swin_base'
    # 训练轮数：固定5000类为300轮
    num_epochs = 300
    # 输出隔离：本地 checkpoints/train_5000
    checkpoint_dir_base = os.getenv("CHECKPOINT_DIR", r"d:\Aic\最终的代码\checkpoints")
    dataset_tag = 'train_5000'
    checkpoint_dir = os.path.join(checkpoint_dir_base, dataset_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_txt_path = os.path.join(checkpoint_dir, "train_log.txt")

    # CUDA/加速策略
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的CUDA，请使用GPU环境运行。")
    print(f"使用设备: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    try:
        torch.cuda.set_per_process_memory_fraction(0.95, device=device)
    except Exception:
        pass

    # 训练数据加载器
    def choose_batch_size(dev, model_tag):
        free_mem, total_mem = torch.cuda.mem_get_info(dev)
        free_gb = free_mem / (1024 ** 3)
        if model_tag.startswith('swin'):
            # 5000类支持更大的批量大小
            if free_gb >= 70: return 1920
            elif free_gb >= 50: return 192
            elif free_gb >= 40: return 128
            elif free_gb >= 20: return 64
            elif free_gb >= 16: return 48
            elif free_gb >= 12: return 32
            elif free_gb >= 8: return 24
            elif free_gb >= 6: return 16
            else: return 12
        return 32

    batch_size = choose_batch_size(device, model_name)
    print(f"动态选择 batch_size: {batch_size}")
    # 仅训练集
    train_loader = get_data_loaders(train_dir=train_dir,
                                    batch_size=batch_size,
                                    mode='train')
    print(f"训练集大小: {len(train_loader.dataset)}")
    # 固定为5000类
    num_classes = selected_class_num
    print(f"类别数量: {num_classes}")

    # 模型
    model = get_model(num_classes, model_name)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    # 使用交叉熵损失函数，标签平滑设为0.15
    label_smoothing = 0.15
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 学习率 / 权重衰减：固定为5000类配置
    base_lr = 3e-4
    learning_rate = base_lr * (batch_size / 32)
    weight_decay = 0.05
    print(f"5000类配置 -> base LR: {base_lr}, 实际 LR: {learning_rate:.3e}, WD: {weight_decay}, Epochs: {num_epochs}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            betas=(0.9, 0.999))

    # 学习率调度：线性预热 + 余弦退火（动态预热）
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = max(2, int(0.1 * num_epochs))
    min_lr = 1e-7

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # 梯度累积：固定为 256，去除 400 类分支依赖
    target_global_batch = 256
    accum_steps = max(1, target_global_batch // batch_size)
    print(f"设置梯度累积步数 accum_steps: {accum_steps}")

    # 断点续训（云路径）
    import glob
    checkpoints = glob.glob(f'{checkpoint_dir}/checkpoint_epoch_*.pth')
    resume_from_epoch = 0
    checkpoint_path = None
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = latest_checkpoint
        resume_from_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
        print(f"找到检查点: {checkpoint_path}, 从第 {resume_from_epoch} 轮继续训练（目标总轮数：{num_epochs}）")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"已加载检查点，从第 {resume_from_epoch} 轮继续训练")

    # 开始训练
    trained_model = train_model(model, train_loader, criterion, optimizer, device,
                               num_epochs=num_epochs, accum_steps=accum_steps,
                               start_epoch=resume_from_epoch, scheduler=scheduler,
                               checkpoint_dir=checkpoint_dir)

    # 保存最终模型到云路径
    torch.save(trained_model.state_dict(), f'{checkpoint_dir}/model_weights.pth')
    print(f"模型已保存到 {checkpoint_dir}/model_weights.pth")

if __name__ == "__main__":
    main()
