import torch
import torchvision.models as models

def disable_inplace_ops(module: torch.nn.Module) -> None:
    """
    Recursively disable inplace operations that can conflict with hooks.
    """
    for child in module.children():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False
        disable_inplace_ops(child)

def load_resnet152(num_classes: int = 1000) -> torch.nn.Module:
    """
    Load a ResNet-152 Model.
    Pretrained weights are not necessary for Phase 1 profiling.
    """

    model = models.resnet152(weights=None)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc_in_features, num_classes)
    disable_inplace_ops(model)
    return model