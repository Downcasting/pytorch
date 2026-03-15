import torch.nn as nn
import timm
from torchvision.ops import MLP

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets import load_dataset, load_from_disk
from .utils import project_to_nullspace


# 2. Model Definition (ViT Small)
class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
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
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1, mode='colored'):
        self.V = V
        self.mode = mode
        if mode == 'clean':
            self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        else:
            self.ds = load_from_disk("./colored_imagenette")[split]
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        if self.mode == 'clean':
            return torch.stack([transform(img) for _ in range(self.V)]), item["label"]
        elif self.mode == 'colored':
            return torch.stack([transform(img) for _ in range(self.V)]), item["label"], item["spurious_label"]

    def __len__(self):
        return len(self.ds)
