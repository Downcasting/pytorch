import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
import importlib.util
import sys

from mnist_lejepa import ColoredMNIST



def visualize_samples(n=5):
    # Load dataset with default transform (which includes normalization)
    ds = ColoredMNIST(root='./data', train=True, V=1)
    
    # Create figure
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    
    # We need to denormalize for visualization if normalization is used
    # Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5] => x = y*std + mean
    inv_normalize = transforms.Normalize(
        mean=[-1.0, -1.0, -1.0], # -mean/std = -0.5/0.5 = -1
        std=[2.0, 2.0, 2.0]      # 1/std = 1/0.5 = 2.0
    )
    
    # Actually simpler: img = img * 0.5 + 0.5
    
    for i in range(n):
        # random index
        idx = np.random.randint(len(ds))
        img, target, color = ds[idx]
        
        # img is [3, 28, 28] Tensor
        # Denormalize
        img_vis = img * 0.5 + 0.5
        
        # Clamp just in case
        img_vis = torch.clamp(img_vis, 0, 1)
        
        # CHW -> HWC for matplotlib
        img_np = img_vis.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Label: {target}\nColor: {color}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('colored_mnist_samples.png')
    print("Saved samples to colored_mnist_samples.png")

if __name__ == '__main__':
    visualize_samples()
