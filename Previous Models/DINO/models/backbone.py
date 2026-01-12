import torchvision.models as models
from torch import nn
import torch
from vit_pytorch import ViT

def get_backbone(name, image_size=224):
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
        backbone = ViTBackbone(
            image_size=image_size,
            patch_size=32,
            num_classes=0,
            dim=384,
            depth=6,
            heads=6,
            mlp_dim=768,
            channels=3
        )
        num_features = 384
    elif name == "vitti":
        backbone = ViTBackbone(
            image_size=image_size,
            patch_size=4,
            num_classes=0,
            dim=192,
            depth=12,
            heads=3,
            mlp_dim=768,
            channels=3
        )
        num_features = 192
    else:
        raise ValueError(f"Unknown backbone model: {name}")
        exit(1)

    backbone.head = nn.Identity()
    return backbone, num_features

class ViTBackbone(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=0, dim=384, depth=6, heads=6, mlp_dim=768, channels=3):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels
        )
    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, dim]  -> CLS token feature
        """
        # patch embedding
        x = self.vit.to_patch_embedding(x)  # [B, num_patches, dim]

        # CLS token 붙이기
        cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, dim]

        # Positional embedding + Dropout
        x = x + self.vit.pos_embedding
        x = self.vit.dropout(x)

        # Transformer layers
        x = self.vit.transformer(x)  # [B, num_patches+1, dim]

        # LayerNorm + CLS token만 추출
        x = self.vit.to_latent(x[:, 0])  # CLS token
        return x