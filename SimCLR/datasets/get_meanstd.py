import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets, transforms
from get_dataset import get_dataset

import os
import sys

def compute_dataset_mean_std(dataset, batch_size=64, num_workers=2, verbose=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = 0.
    std = 0.
    total_images = 0

    # 데이터를 모두 모으기 위한 리스트
    all_data = []

    loop = tqdm(loader, desc="Gathering dataset images") if verbose else loader

    for data, _ in loop:
        all_data.append(data)
        total_images += data.size(0)

    # 모든 배치 합치기 (N, C, H, W)
    all_data = torch.cat(all_data, dim=0)

    # 채널별 평균, std 계산 (전체 데이터셋 픽셀 기준)
    mean = all_data.mean(dim=[0, 2, 3])
    std = all_data.std(dim=[0, 2, 3])

    return mean.tolist(), std.tolist()

if __name__ == "__main__":

    dataset_name = sys.argv[1]
    transform = transforms.ToTensor()
    dataset = get_dataset(name=dataset_name, transform=transform, root="./../../data")

    mean, std = compute_dataset_mean_std(dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
