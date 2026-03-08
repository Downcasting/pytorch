import torch.nn as nn
import timm
from torchvision.ops import MLP

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset, load_from_disk

from .utils import project_to_nullspace


# 2. Model Definition (ViT Tiny)
class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        # patch_size=4, img_size=32 for CIFAR-10. num_classes=0 to extract features only
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=32,             # 이미지 사이즈 32로 변경
            patch_size=4,            # 패치 사이즈 4로 변경
        )
        num_ftrs = self.backbone.num_features # typically 192 for ViT-Tiny
        self.proj = MLP(num_ftrs, [512, 512, proj_dim], norm_layer=nn.BatchNorm1d)
        self.eigenvecs = None

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        if self.eigenvecs is not None:
            emb = project_to_nullspace(emb, self.eigenvecs)
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

    def update_eigenvecs(self, eigvecs):
        self.eigenvecs = eigvecs.detach().to(next(self.parameters()).device)

    
# 3. Dataset Definition
class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1, mode='colored'):
        self.V = V
        self.mode = mode
        if mode == 'clean':
            self.ds = load_dataset("cifar10", split=split)
        else:
            self.ds = load_from_disk("./colored_cifar10")[split]
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(32, scale=(0.2, 1.0)),
                # v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                # v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]),
                # v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(32),
                v2.CenterCrop(32),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["img"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        if self.mode == 'clean':
            return torch.stack([transform(img) for _ in range(self.V)]), item["label"]
        elif self.mode == 'colored':
            return torch.stack([transform(img) for _ in range(self.V)]), item["label"], item["spurious_label"]

    def __len__(self):
        return len(self.ds)
