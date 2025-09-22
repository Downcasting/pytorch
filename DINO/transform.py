import torchvision.transforms as T
from PIL import ImageFilter
import random

import torch

class GaussianBlur:
    """약한/강한 Gaussian Blur 커스터마이징."""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(*self.sigma)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

def get_dino_transform(cfg, global_crop=True, blur_strength=None):
    # 공통 변형
    jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)

    if global_crop:
        scale = (0.4, 1.0)  # Global crop의 스케일
    else:
        scale = (0.15, 0.5)

    if cfg['transform']['gaussian_blur']:
        if blur_strength == 'Strong':
            blur = 2.0
        elif blur_strength == "Weak":
            blur = 0.5
        else:
            print("Invalid blur strength. Using default [0.1, 2.0].")
            blur = 2.0
    else:
        blur = -1.0  # Gaussian Blur 사용 안함

    transform = T.Compose([
        T.RandomResizedCrop(cfg['dataset']['input_size'], scale=scale),
        T.RandomHorizontalFlip(),
        T.RandomApply([jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(sigma=[0.1, blur]) if blur > 0 else T.Identity(),
        T.ToTensor(),
        T.Normalize(mean=cfg['transform']['normalize']['mean'], std=cfg['transform']['normalize']['std'])
    ])
    return transform

class DINOTransform:
    """global, local를 만들어주는 transform 클래스"""
    def __init__(self, cfg):
        self.global_transforms = [get_dino_transform(cfg, global_crop=True, blur_strength='Weak'), 
                                  get_dino_transform(cfg, global_crop=True, blur_strength='Strong')]
        self.local_transforms = [get_dino_transform(cfg, global_crop=False, blur_strength='Strong') for _ in range(4)]

    def __call__(self, x):
        globals_ = [t(x) for t in self.global_transforms]
        locals_ = [t(x) for t in self.local_transforms]
        return globals_ + locals_
    
def dino_collate_fn(batch):
    # batch: [(views, label), ...]
    views_batch, labels = zip(*batch)

    # global: 앞 2개 / local: 나머지
    globals_batch = [v[:2] for v in views_batch]
    locals_batch  = [v[2:] for v in views_batch]

    # transpose crop-wise
    globals_batch = list(zip(*globals_batch))  # [(B개의 global1), (B개의 global2)]
    locals_batch  = list(zip(*locals_batch))   # [(B개의 local1), (B개의 local2), ...]

    # stack each crop into (B, C, H, W)
    globals_batch = [torch.stack(crops) for crops in globals_batch]
    locals_batch  = [torch.stack(crops) for crops in locals_batch]

    # global 먼저, local 뒤에
    all_crops = globals_batch + locals_batch

    return all_crops, torch.tensor(labels)


class DINOEvalTransform:
    """평가용 transform 클래스"""
    def __init__(self, cfg):
        self.transform = get_dino_transform(cfg, global_crop=True, blur_strength='Weak')

    def __call__(self, x):
        return self.transform(x)
