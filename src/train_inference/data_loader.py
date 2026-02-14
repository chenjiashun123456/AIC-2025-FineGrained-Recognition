import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class WebFG400Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录
            mode (string): 'train' 或 'test'
            transform (callable, optional): 应用于图像的转换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            # 对于训练集，root_dir 直接指向 train_cleaned 目录
            self.data_dir = root_dir
            # 获取所有类别文件夹
            self.classes = sorted([d for d in os.listdir(self.data_dir) 
                                  if os.path.isdir(os.path.join(self.data_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            # 收集所有图像路径和标签
            self.images = []
            self.labels = []
            for cls_name in self.classes:
                cls_dir = os.path.join(self.data_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    ext = os.path.splitext(img_name)[1].lower()
                    if ext in {'.jpg', '.jpeg', '.png'}:
                        self.images.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls_name])
        else:
            self.data_dir = root_dir
            self.images = [os.path.join(self.data_dir, img_name) for img_name in os.listdir(self.data_dir)
                           if os.path.splitext(img_name)[1].lower() in {'.jpg', '.jpeg', '.png'}]
            self.labels = None
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            if self.mode == 'train':
                return image, self.labels[idx]
            else:
                # 对于测试集，我们返回图像和文件名（用于提交结果）
                img_name = os.path.basename(img_path)
                return image, img_name
        except (OSError, IOError) as e:
            # 如果图像损坏，打印错误信息并返回一个替代图像
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 创建一个空白的RGB图像作为替代
            dummy_img = Image.new('RGB', (256, 256), color=(0, 0, 0))
            
            if self.transform:
                dummy_img = self.transform(dummy_img)
                
            if self.mode == 'train':
                return dummy_img, self.labels[idx]
            else:
                img_name = os.path.basename(img_path)
                return dummy_img, img_name

# 方法：get_data_loaders
from torch.utils.data import DataLoader

def get_data_loaders(train_dir=None, test_dir=None, batch_size=32, mode='train'):
    """
    创建数据加载器
    """
    # 修改原因：
    # - ConvNeXt 官方推荐 224 输入分辨率；降低显存峰值、提升吞吐，同时与预训练配置更匹配
    # 预期效果：
    # - 在 4090 24GB 显存下更稳的训练，且能覆盖 400/5000 类两场景
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, test_loader = None, None

    if mode in ['train', 'all'] and train_dir:
        train_dataset = WebFG400Dataset(root_dir=train_dir, mode='train', transform=train_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    if mode in ['test', 'all'] and test_dir:
        test_dataset = WebFG400Dataset(root_dir=test_dir, mode='test', transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
    if mode == 'all':
        return train_loader, test_loader
    elif mode == 'train':
        return train_loader
    elif mode == 'test':
        return test_loader
