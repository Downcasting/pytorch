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

def get_dino_transform(cfg, global_crop=True):
    # 공통 변형
    normalize = T.Normalize(mean=cfg['transform']['normalize']['mean'],
                            std=cfg['transform']['normalize']['std'])
    jitter = T.ColorJitter(0.8*cfg['transform']['jitter_strength'],
                           0.8*cfg['transform']['jitter_strength'],
                           0.8*cfg['transform']['jitter_strength'],
                           0.2*cfg['transform']['jitter_strength'])

    transform = T.Compose([
        T.RandomResizedCrop(cfg['dataset']['input_size'], scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        T.RandomApply([jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur() if cfg['transform']['gaussian_blur'] else T.Identity(),
        T.ToTensor(),
        normalize
    ])
    return transform

class DINODualTransform:
    """view1, view2를 만들어주는 transform 클래스"""
    def __init__(self, cfg):
        self.view1 = get_dino_transform(cfg)
        self.view2 = get_dino_transform(cfg)

    def __call__(self, x):
        return self.view1(x), self.view2(x)
