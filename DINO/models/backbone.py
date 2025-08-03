import torchvision.models as models
from torch import nn
from vit_pytorch import ViT

def get_backbone(name, image_size=32):
    backbone = None
    name = str(name).lower()
    if name == "resnet18":
        backbone = models.resnet18()
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == "resnet50":
        backbone = models.resnet50()
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == "vits16":
        backbone = ViT(
            image_size=image_size,
            patch_size=4,
            num_classes=0,
            dim=384,
            depth=6,
            heads=6,
            mlp_dim=768
        )
        num_features = 384
    else:
        raise ValueError(f"Unknown backbone model: {name}")
        exit(1)
    return backbone, num_features