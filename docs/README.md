# 项目参考文献与资料

本项目参考了多篇关于**网络监督细粒度图像识别**与**噪声标签学习 (Noisy Label Learning)** 的前沿论文。以下是核心参考文献及其要点总结。

## 核心论文

### 1. WebFG 数据集与 Peer-learning
- **文件**: `web.txt`
- **对应论文**: *Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach*
- **主要内容**:
    - **数据集来源**: 介绍了 WebFG-496 和 WebiNat-5089 数据集的构建过程（Bing Image Search, Flickr）。
    - **核心挑战**: 跨域噪声（非目标物体）与类内噪声（标签错误）。
    - **方法**: 提出了 **Peer-learning**（同行学习），利用双网络互相筛选样本（共识集 vs 分歧集）来对抗噪声，是 Co-teaching 的重要前身。

### 2. NPN: Noisy Label Learning
- **文件**: `2312.txt`
- **对应论文**: *NPN* (具体标题待补，关于 Partial Label Learning + Negative Learning)
- **主要内容**:
    - **核心思想**: 将噪声标签问题转化为 **Partial Label Learning (PLL)** 和 **Negative Learning (NL)**。
    - **策略**: 不直接“纠错”，而是划定“候选集”（可能对）和“互补集”（肯定错）。
    - **优势**: 避免了直接剔除样本带来的信息损失，利用所有样本进行鲁棒训练。

### 3. SED: 自适应与类平衡选择
- **文件**: `2407.txt`
- **对应论文**: *Foster Adaptivity and Balance in Learning with Noisy Labels*
- **主要内容**:
    - **改进点**: 针对 Small Loss Trick 的缺陷（阈值固定、忽略长尾分布）进行升级。
    - **SCS (Smart Selection)**: 基于置信度的动态阈值，且对每个类别有独立的阈值（类平衡）。
    - **SCR (Correction & Re-weighting)**: 使用 Teacher 模型修正噪声标签，并根据置信度重加权。
    - **应用**: 非常适合本项目的长尾分布与细粒度场景。

## 资料文件说明

`summaries/` 目录下包含了上述论文的详细中文解读文本，建议深入阅读以优化模型训练策略（如引入 SCS/SCR 模块）。
