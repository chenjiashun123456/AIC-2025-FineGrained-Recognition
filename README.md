# AIC-2025 网络监督细粒度图像识别挑战赛 (SwinBase)

本仓库包含 AIC-2025 网络监督细粒度图像识别挑战赛的代码库。它实现了使用 Swin Transformer 和其他最先进模型的训练和推理流程。

## 项目结构

代码组织为以下模块：

- **`src/train_inference/`**:
  - `model.py`: 模型定义与工厂。支持 Swin Transformer V2 Base, Swin Base, ConvNeXt V2 Large, ResNet50, VGG16 和 DenseNet121。
  - `data_loader.py`: 自定义 `Dataset` 和 `DataLoader` 实现 (`WebFG400Dataset`)，处理图像加载、变换以及损坏图像的错误处理。
  - `train-400.py`: 针对 400 类场景优化的训练脚本。具有混合精度训练 (AMP)、梯度累积和 Top-k 检查点保留功能。
  - `train-5000.py`: 针对 5000 类场景优化的训练脚本。包含基于 GPU 显存的动态批量大小调整。

- **`src/preprocessing/`**:
  - `dedup.py`: 图像去重脚本。使用感知哈希 (pHash) 和中间特征（从 ResNet50 layer 3 提取）的余弦相似度来识别和过滤重复图像。

## 主要特性

- **先进模型**: 利用 `timm` 库支持 Swin Transformers 和 ConvNeXt 模型。
- **鲁棒训练**: 实现梯度累积、混合精度训练 (`torch.cuda.amp`) 和动态学习率调度（带预热的余弦退火）。
- **数据质量**: 包含用于检测和删除重复图像的工具，以提高数据集质量。
- **高效性**: 动态批量大小调整和优化的数据加载 worker。

## 使用方法

### 训练

训练模型（以 5000 类为例）：

```bash
python src/train_inference/train-5000.py --train_dir /path/to/dataset
```

### 数据去重

运行去重过程：

```bash
python src/preprocessing/dedup.py --train_dir /path/to/dataset
```
