import torch
import datasets
from datasets import DatasetDict
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import os

# 1. Dataset Definition
ds_train = datasets.load_dataset("cifar10", split="train")
ds_test = datasets.load_dataset("cifar10", split="test")

box_size = 3
# 8 positions for 32x32 image with distance 3 from edge/corner
# left = 3, top = 3
# right = 32 - 3(dist) - 3(box_size) = 26
# bottom = 32 - 3(dist) - 3(box_size) = 26
# center = 14
positions = [
    (3, 3),   # Left-Up
    (14, 3),  # Up
    (26, 3),  # Right-Up
    (26, 14), # Right
    (26, 26), # Right-Down
    (14, 26), # Down
    (3, 26),  # Left-Down
    (3, 14)   # Left
]

colors = torch.tensor([
    [255, 0, 0],       # 0
    [0, 255, 0],       # 1
    [0, 0, 255],       # 2
    [180, 180, 0],     # 3
    [180, 0, 180],     # 4
    [0, 180, 180],     # 5
    [228, 114, 0],     # 6
    [228, 0, 114],     # 7
    [114, 228, 0],     # 8
    [0, 114, 228]      # 9
], dtype=torch.float32)

def add_box(example):
    img = example["img"].convert("RGB")
    label = example["label"]

    draw = ImageDraw.Draw(img)
    
    # 2. Spurious Correlation (70% correlated)
    if np.random.rand() < 0.7:
        color_idx = label
    else:
        possible_colors = list(range(10))
        possible_colors.remove(label)
        color_idx = np.random.choice(possible_colors)
        
    box_color = tuple(colors[color_idx].int().tolist())
    
    for (x, y) in positions:
        draw.rectangle([x, y, x + box_size, y + box_size], fill=box_color)

    example["img"] = img 
    example["spurious_label"] = int(color_idx)
    return example

def main():
    # 3. Apply to dataset
    modified_ds_train = ds_train.map(add_box, batched=False)
    modified_ds_test = ds_test.map(add_box, batched=False)

    save_path = './colored_cifar10'

    dataset_dict = DatasetDict({
        "train": modified_ds_train,
        "test": modified_ds_test
    })

    dataset_dict.save_to_disk(save_path)

    # 4. Check results
    print("Original Dataset:", ds_train)
    print("Modified Dataset:", modified_ds_train)

    os.makedirs('generated_samples', exist_ok=True)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    
    indices = np.random.choice(len(modified_ds_train), 5, replace=False)
    
    for i, idx in enumerate(indices):
        sample = modified_ds_train[int(idx)]
        img = sample['img']
        lbl = sample['label']
        spurious_lbl = sample['spurious_label']
        
        img.save(f"generated_samples/sample_{i}_label_{lbl}_spurious_{spurious_lbl}.png")

        axs[i].imshow(img)
        axs[i].set_title(f"Lbl: {lbl}, Spur: {spurious_lbl}")
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('generated_samples/combined_samples.png')
    plt.show()

if __name__ == "__main__":
    main()