import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets, transforms
from get_dataset import get_dataset

import os
import sys

def compute_dataset_mean_std(dataset, batch_size=32, num_workers=0, verbose=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_channels = 3  # 기본 RGB, 흑백이면 1
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    total_images = 0

    loop = tqdm(loader, desc="Computing mean/std") if verbose else loader

    for data, _ in loop:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()
if __name__ == "__main__":

    dataset_name = sys.argv[1]
    transform = transforms.ToTensor()
    dataset = get_dataset(name=dataset_name, transform=transform, root="./../../data")

    mean, std = compute_dataset_mean_std(dataset)
    mean = [round(m, 4) for m in mean]
    std = [round(s, 4) for s in std]
    print(f"Mean: {mean}")
    print(f"Std: {std}")
