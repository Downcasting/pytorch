# dataset_utils.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets, transforms
from get_dataset import get_dataset

def compute_dataset_mean_std(dataset, batch_size=64, num_workers=2, verbose=True):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = 0.
    std = 0.
    total_samples = 0

    loop = tqdm(loader, desc="Computing mean/std") if verbose else loader

    for data, _ in loop:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)

        mean += data.mean(2).sum(0)  # (C,)
        std += data.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    # Example usage with CIFAR10 dataset

    transform = transforms.ToTensor()
    dataset = get_dataset(name='stl10', transform=transform, root="./../../data")

    print(compute_dataset_mean_std(dataset))