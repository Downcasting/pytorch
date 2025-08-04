import torchvision.transforms as T
from PIL import ImageFilter
import random

class GaussianBlur:
    """약한/강한 Gaussian Blur 커스터마이징."""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(*self.sigma)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

def get_dino_transform(cfg, global_crop=True, blur_strength=None):
    # 공통 변형
    normalize = T.Normalize(mean=cfg['transform']['normalize']['mean'],
                            std=cfg['transform']['normalize']['std'])
    jitter = T.ColorJitter(0.8*cfg['transform']['jitter_strength'],
                           0.8*cfg['transform']['jitter_strength'],
                           0.8*cfg['transform']['jitter_strength'],
                           0.2*cfg['transform']['jitter_strength'])

    if cfg['transform']['gaussian_blur']:
        if blur_strength == 'Strong':
            blur = GaussianBlur(sigma=[0.1, 2.0])
        elif blur_strength == "Weak":
            blur = GaussianBlur(sigma=[0.1, 0.5])
        else:
            print("Invalid blur strength. Using default [0.1, 2.0].")
            blur = GaussianBlur(sigma=[0.1, 2.0])
    else:
        blur = T.Identity()  # Gaussian Blur 사용 안함

    transform = T.Compose([
        T.RandomResizedCrop(cfg['dataset']['input_size'], scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        T.RandomApply([jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        blur,
        T.ToTensor(),
        normalize
    ])
    return transform

class DINOTransform:
    """global, local를 만들어주는 transform 클래스"""
    def __init__(self, cfg):
        self.global_transforms = [get_dino_transform(cfg, global_crop=True, blur_strength='Strong') for _ in range(2)]
        self.local_transforms = [get_dino_transform(cfg, global_crop=False, blur_strength='Weak') for _ in range(6)]

    def __call__(self, x):
        globals_ = [t(x) for t in self.global_transforms]
        locals_ = [t(x) for t in self.local_transforms]
        return globals_, locals_

class DINOEvalTransform:
    """평가용 transform 클래스"""
    def __init__(self, cfg):
        self.transform = get_dino_transform(cfg, global_crop=False, blur_strength='Weak')

    def __call__(self, x):
        return self.transform(x)