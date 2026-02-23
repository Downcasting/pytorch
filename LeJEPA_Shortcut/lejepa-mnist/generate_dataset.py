import torch
import torch.nn as nn
from torchvision import datasets
import os
import sys

def get_color_palette():
    # 10 distinct colors (RGB)
    return torch.tensor([
        [1.0, 0.0, 0.0], # 0: Red
        [0.0, 1.0, 0.0], # 1: Green
        [0.0, 0.0, 1.0], # 2: Blue
        [1.0, 1.0, 0.0], # 3: Yellow
        [1.0, 0.0, 1.0], # 4: Magenta
        [0.0, 1.0, 1.0], # 5: Cyan
        [1.0, 0.5, 0.0], # 6: Orange
        [0.5, 0.0, 1.0], # 7: Purple
        [0.5, 1.0, 0.0], # 8: Lime
        [0.0, 0.5, 1.0], # 9: Azure
    ])

def generate_colored_mnist(root='./data', train_corr=0.9, test_corr=0.1):
    os.makedirs(root, exist_ok=True)
    
    palette = get_color_palette()
    
    for split in ['train', 'test']:
        train = (split == 'train')
        mnist = datasets.MNIST(root, train=train, download=True)
        
        # Targets are 0-9
        targets = mnist.targets.clone()
        
        # Define correlation
        corr = train_corr if train else test_corr
        
        # Assign colors
        # With probability `corr`, color = target
        # With probability `1-corr`, color = random other class
        
        colors = targets.clone()
        rand_mask = torch.rand(len(targets)) > corr
        
        # To ensure the random color is different from the target:
        # Pick a random offset in [1, 9] add to target, take modulo 10
        random_offset = torch.randint(1, 10, (len(targets),))
        colors[rand_mask] = (targets[rand_mask] + random_offset[rand_mask]) % 10
        
        # Prepare Images
        images = mnist.data.float() / 255.0
        # Stack to 3 channels: [R, G, B]
        colored_images = torch.zeros(len(images), 3, 28, 28)
        
        # Vectorized coloring
        for c in range(10):
            mask = (colors == c)
            if mask.sum() > 0:
                # Add color c to images with color label c
                # Image is grayscale (1 channel), Palette has shape (3,)
                # Result should be (N, 3, 28, 28)
                color_vec = palette[c].view(1, 3, 1, 1)
                colored_images[mask] = images[mask].unsqueeze(1) * color_vec
        
        # Save
        dataset = {
            'images': colored_images,
            'targets': targets.long(),
            'colors': colors.long()
        }
        torch.save(dataset, os.path.join(root, f'colored_mnist_{split}.pt'))
        print(f"Saved {split} dataset with correlation {corr} to {os.path.join(root, f'colored_mnist_{split}.pt')}")

if __name__ == '__main__':
    # Default values
    train_corr = 0.9
    test_corr = 0.1
    
    # Parse generic argv "key=value"
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=')
            if key == 'train_correlation':
                train_corr = float(value)
            elif key == 'test_correlation':
                test_corr = float(value)
                
    generate_colored_mnist(train_corr=train_corr, test_corr=test_corr)
