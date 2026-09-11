"""构建预训练 ResNet18，并仅微调 layer4 与分类头。"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """构建模型；训练时必须启用预训练权重。"""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError("无法加载 ResNet18 预训练权重，请检查网络连接") from exc
        raise

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

