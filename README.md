# AIC-2025 Fine-Grained Image Recognition Challenge (SwinBase)

This repository contains the codebase for the AIC-2025 Network Supervised Fine-Grained Image Recognition Challenge. It implements training and inference pipelines using Swin Transformer and other state-of-the-art models.

## Project Structure

The code is organized into the following modules:

- **`src/train_inference/`**:
  - `model.py`: Model definition and factory. Supports Swin Transformer V2 Base, Swin Base, ConvNeXt V2 Large, ResNet50, VGG16, and DenseNet121.
  - `data_loader.py`: Custom `Dataset` and `DataLoader` implementation (`WebFG400Dataset`) handling image loading, transformations, and error handling for corrupted images.
  - `train-400.py`: Training script optimized for 400-class scenarios. Features mixed precision training (AMP), gradient accumulation, and top-k checkpoint preservation.
  - `train-5000.py`: Training script optimized for 5000-class scenarios. Includes dynamic batch size adjustment based on GPU memory.

- **`src/preprocessing/`**:
  - `dedup.py`: Image deduplication script. Uses Perceptual Hash (pHash) and Cosine Similarity of intermediate features (extracted from ResNet50 layer 3) to identify and filter duplicate images.

## Key Features

- **Advanced Models**: Utilizes `timm` for Swin Transformers and ConvNeXt models.
- **Robust Training**: Implements gradient accumulation, mixed precision training (`torch.cuda.amp`), and dynamic learning rate scheduling (Cosine Annealing with Warmup).
- **Data Quality**: Includes tools for detecting and removing duplicate images to improve dataset quality.
- **Efficiency**: Dynamic batch sizing and optimized data loading workers.

## Usage

### Training

To train the model (example for 5000 classes):

```bash
python src/train_inference/train-5000.py --train_dir /path/to/dataset
```

### Data Deduplication

To run the deduplication process:

```bash
python src/preprocessing/dedup.py --train_dir /path/to/dataset
```
