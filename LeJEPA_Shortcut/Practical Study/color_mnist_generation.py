import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from tqdm import tqdm
import os

class ExtentBiasColoredMNIST(Dataset):
    def __init__(self, root='./data', train=True, object_ratio=0.05, correlation=0.7, download=True):
        """
        논문 Section 7.1 & Appendix D.2 설정을 따르는 Custom Dataset
        
        Args:
            root: 데이터 저장 경로
            train: 학습용(True) / 테스트용(False)
            object_ratio: 전체 이미지 대비 숫자(객체)가 차지하는 면적 비율 (0.05, 0.10, 0.15)
            correlation: 배경색과 라벨(숫자)의 상관관계 (논문 설정: 0.7)
        """
        self.mnist = datasets.MNIST(root, train=train, download=download)
        self.object_ratio = object_ratio
        self.correlation = correlation
        
        # 논문 D.2에 명시된 10가지 배경색 (RGB)
        self.colors = torch.tensor([
            [255, 0, 0],       # 0
            [0, 255, 0],       # 1
            [0, 0, 255],       # 2
            [180, 180, 0],     # 3 (소수점 반올림)
            [180, 0, 180],     # 4
            [0, 180, 180],     # 5
            [228, 114, 0],     # 6
            [228, 0, 114],     # 7
            [114, 228, 0],     # 8
            [0, 114, 228]      # 9
        ], dtype=torch.float32) / 255.0  # 0~1 사이로 정규화
        
        # 전체 이미지 크기 (논문 D.2: 48x48)
        self.image_size = 48
        
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        
        # 1. 배경색 결정 (Spurious Correlation)
        # 70% 확률로 라벨과 일치하는 색, 30% 확률로 랜덤 색
        if np.random.rand() < self.correlation:
            color_idx = label
        else:
            # 라벨을 제외한 나머지 색 중에서 랜덤 선택
            possible_colors = list(range(10))
            possible_colors.remove(label)
            color_idx = np.random.choice(possible_colors)
            
        bg_color = self.colors[color_idx]
        
        # 2. Extent Bias 구현 (숫자 크기 조절)
        # Target Area = 48 * 48 * ratio
        # 숫자 이미지의 한 변의 길이(scale) 계산
        target_area = (self.image_size ** 2) * self.object_ratio
        scale = int(np.sqrt(target_area))
        scale = max(4, min(scale, 40)) # 너무 작거나 크지 않게 클리핑
        
        # MNIST 원본(28x28)을 계산된 scale 크기로 리사이즈
        digit_transform = transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.ToTensor()
        ])
        digit_tensor = digit_transform(img) # (1, scale, scale), 값은 0~1
        
        # 3. 이미지 합성 (Colored Background + Gray Digit)
        # 48x48 배경 생성
        full_img = torch.ones((3, self.image_size, self.image_size)) * bg_color.view(3, 1, 1)
        
        # 숫자를 중앙에 배치
        start_x = (self.image_size - scale) // 2
        start_y = (self.image_size - scale) // 2
        
        # 마스크 생성 (숫자가 있는 부분)
        # 숫자가 있는 부분(digit_tensor > 0.1)은 원래 숫자 색(흰색/회색)으로,
        # 없는 부분은 배경색을 유지.
        # 일반적으로 Colored MNIST는 '배경색 위에 흰색 숫자' 혹은 '숫자에 색칠' 두 가지가 있는데,
        # 논문 맥락상 "배경 힌트"가 중요하므로 배경 전체를 칠하고 숫자를 위에 얹습니다.
        
        # 숫자가 얹혀질 영역 추출
        roi = full_img[:, start_y:start_y+scale, start_x:start_x+scale]
        
        # 숫자 마스크 (0~1)
        mask = digit_tensor.repeat(3, 1, 1)
        
        # 합성: (배경 * (1-마스크)) + (숫자 * 마스크)
        # 숫자는 흰색(1.0)이라고 가정
        digit_color = 1.0 
        composed_roi = roi * (1 - mask) + digit_color * mask
        
        full_img[:, start_y:start_y+scale, start_x:start_x+scale] = composed_roi
        
        return full_img, label


def save_colored_mnist(dataset, save_dir, split):
    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    images = []
    labels = []

    for x, y in tqdm(loader, desc=f"Saving {split}"):
        images.append(x)
        labels.append(y)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    torch.save(images, os.path.join(save_dir, f"{split}_images.pt"))
    torch.save(labels, os.path.join(save_dir, f"{split}_labels.pt"))

    print(f"[✔] {split} saved:", images.shape, labels.shape)



# --- 사용 예시 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. 데이터셋 생성 (비율 0.05, 0.10, 0.15 테스트)
    ratios = [0.05, 0.10, 0.15]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    for i, r in enumerate(ratios):
        dataset = ExtentBiasColoredMNIST(train=True, object_ratio=r, correlation=0.7)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        
        images, labels = next(iter(loader))
        
        for j in range(5):
            img = images[j].permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"Ratio {r}, Label {labels[j].item()}")
            axes[i, j].axis('off')
            

    
    train_dataset = ExtentBiasColoredMNIST(
        train=True, object_ratio=0.05, correlation=0.7
    )
    test_dataset = ExtentBiasColoredMNIST(
        train=False, object_ratio=0.05, correlation=0.7
    )

    save_colored_mnist(train_dataset, "./colored_mnist", "train")
    save_colored_mnist(test_dataset, "./colored_mnist", "test")
