import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 모델 클래스 정의 (이전 코드에서 사용한 모델과 동일해야 함)
class SimpleUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        
        self.conv0 = torch.nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = torch.nn.ModuleList([torch.nn.Conv2d(down_channels[i], down_channels[i+1], 3, padding=1) 
                                          for i in range(len(down_channels)-1)])
        self.ups = torch.nn.ModuleList([torch.nn.ConvTranspose2d(up_channels[i], up_channels[i+1], 4, 2, 1) 
                                        for i in range(len(up_channels)-1)])
        self.output = torch.nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x):
        x = self.conv0(x)
        for down in self.downs:
            x = down(x)
        for up in self.ups:
            x = up(x)
        return self.output(x)

# 저장된 모델 불러오기
def load_model(model_path, device):
    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 이미지 변환 함수
def preprocess_image(image_path, img_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# 결과 시각화
def show_images(original, restored):
    original = (original.squeeze(0).permute(1, 2, 0) + 1) / 2  # Normalize to [0, 1]
    restored = (restored.squeeze(0).permute(1, 2, 0) + 1) / 2  # Normalize to [0, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original.numpy())
    axes[0].set_title("Original Image")
    axes[1].imshow(restored.detach().numpy())
    axes[1].set_title("Restored Image")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./diffusion_model_50.pth"  # 불러올 모델 파일
    image_path = "./../data/cars_test/00001.jpg"  # 테스트할 이미지 파일 경로

    model = load_model(model_path, device)
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        restored_image = model(image)
    
    show_images(image.cpu(), restored_image.cpu())