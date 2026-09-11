"""验证 ResNet18 分类头和可训练层配置。"""

import torch.nn as nn

from catdog_classifier.model import build_resnet18


def test_model_only_unfreezes_layer4_and_head() -> None:
    model = build_resnet18(num_classes=2, pretrained=False)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 2
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "fc.weight" in trainable_names
    assert "fc.bias" in trainable_names
    assert any(name.startswith("layer4.") for name in trainable_names)
    assert not any(name.startswith("layer3.") for name in trainable_names)

