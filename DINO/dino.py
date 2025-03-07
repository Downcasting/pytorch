import torch

# DINO 사전 학습 모델 로드 (ViT-Small/16)
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# 모델 구조 출력
print(model)