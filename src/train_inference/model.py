import torch
import torch.nn as nn
import torchvision.models as models
import os
import timm

# 函数：get_model
def get_model(num_classes, model_name='swin_base'):
    """
    获取预训练模型并修改最后的分类层
    
    Args:
        num_classes (int): 类别数量
        model_name (str): 模型名称，如'resnet50', 'vgg16', 'densenet121'等
        
    Returns:
        model: 修改后的预训练模型
    """
    if model_name == 'resnet50':
        # 加载预训练的ResNet50模型
        model = models.resnet50(weights='IMAGENET1K_V1')
        # 修改最后的全连接层以匹配我们的类别数
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # 新增 Swin-Transformer V2 Base（ImageNet-1k 预训练）
    elif model_name == 'swinv2_base':
        model = timm.create_model('swinv2_base_window8_256', pretrained=True, num_classes=num_classes)
        return model
    
    # 新增 Swin-Transformer Base（已有）
    elif model_name == 'swin_base':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
        return model
    # 修复后的 ConvNeXt V2-Large 分支：本地 1k-only 权重优先，失败回退 V1 Large (1k-only)
    elif model_name == 'convnextv2_large.fcmae_ft_in1k':
        # 单次实例化，避免重复创建与强制联网
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        return model
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model
