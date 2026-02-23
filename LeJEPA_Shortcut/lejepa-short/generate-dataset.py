import torch
import datasets
from datasets import DatasetDict
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt # import 위치 이동

import os

# 1. Dataset Definition
ds_train = datasets.load_dataset("frgfm/imagenette", "160px", split="train")
ds_test = datasets.load_dataset("frgfm/imagenette", "160px", split="validation")

box_count = 64
box_size = 8
# image_size = 128  <-- 제거: 이미지 크기는 실제 이미지에서 가져오는 게 안전합니다.

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
], dtype=torch.float32) # / 255.0 부분은 아래에서 처리하거나 여기서 하고 아래에서 빼거나 통일 필요
# 여기서는 0~255 int값으로 바로 쓰는게 PIL에 그리기 편하므로 정규화 제거하고 아래 로직 수정함

def add_box(example):
    
    img = example["image"].convert("RGB")
    label = example["label"]
    w, h = img.size  # 이미지 크기 동적 확인 (160x160)

    draw = ImageDraw.Draw(img)
    
    # 2. 배경색 결정 (Spurious Correlation)
    if np.random.rand() < 0.7:
        color_idx = label
    else:
        possible_colors = list(range(10))
        possible_colors.remove(label)
        color_idx = np.random.choice(possible_colors)
        
    # colors 텐서에서 색상을 가져와 정수 튜플로 변환 (PIL용)
    # 위에서 /255.0을 뺐으므로 바로 int로 변환
    box_color = tuple(colors[color_idx].int().tolist())
    
    # 1. 격자의 행/열 개수 계산 (예: 16개 -> 4x4)
    grid_side = int(np.ceil(np.sqrt(box_count)))
    
    # 2. 격자 한 칸의 크기 (Stride)
    step_x = w // grid_side
    step_y = h // grid_side
    
    # 3. 격자 칸 내에서 박스를 중앙 정렬하기 위한 오프셋
    offset_x = (step_x - box_size) // 2
    offset_y = (step_y - box_size) // 2

    for i in range(box_count):
        # 현재 박스가 몇 번째 행, 몇 번째 열인지 계산
        row = i // grid_side
        col = i % grid_side
        
        # 좌표 계산: (격자 시작점) + (중앙 정렬 오프셋)
        x = (col * step_x) + offset_x
        y = (row * step_y) + offset_y
        
        # 박스 그리기
        draw.rectangle([x, y, x + box_size, y + box_size], fill=box_color)

    # [핵심 수정] 튜플이 아닌, 수정된 example 딕셔너리 자체를 반환해야 합니다.
    example["image"] = img 
    example["spurious_label"] = int(color_idx)
    return example

def main():
    # 3. 데이터셋에 적용
    # num_proc을 사용하면 멀티프로세싱으로 속도가 빨라집니다 (선택사항)
    modified_ds_train = ds_train.map(add_box, batched=False)
    modified_ds_test = ds_test.map(add_box, batched=False) # 테스트용은 잠시 주석

    save_path = './colored_imagenette'

    # 하나의 DatasetDict 객체로 묶음
    dataset_dict = DatasetDict({
        "train": modified_ds_train,
        "validation": modified_ds_test  # 위 코드에서 처리한 test셋 변수
    })

    # 한 번에 저장
    dataset_dict.save_to_disk(save_path)

    # 4. 결과 확인
    print("Original Dataset:", ds_train)
    print("Modified Dataset:", modified_ds_train)

    # 샘플 이미지 저장용 폴더 생성
    os.makedirs('generated_samples', exist_ok=True)

    # 샘플 이미지 5장 확인 및 저장
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    
    # 랜덤하게 섞어서 5장 뽑기 (수정된 데이터셋에서)
    indices = np.random.choice(len(modified_ds_train), 5, replace=False)
    
    for i, idx in enumerate(indices):
        sample = modified_ds_train[int(idx)] # numpy int -> python int
        img = sample['image']
        lbl = sample['label']
        
        # 개별 이미지 저장 (프레젠테이션용)
        img.save(f"generated_samples/sample_{i}_label_{lbl}.png")

        axs[i].imshow(img)
        axs[i].set_title(f"Label: {lbl}")
        axs[i].axis('off')
        
    # 전체 subplot 저장
    plt.tight_layout()
    plt.savefig('generated_samples/combined_samples.png')
    plt.show()

if __name__ == "__main__":
    main()