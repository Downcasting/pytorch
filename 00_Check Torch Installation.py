import torch

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tensor 생성
x = torch.rand(3).to(device)
print(f"Tensor on {device}: {x}")

# GPU 메모리 사용량 확인
print(f"Allocated GPU memory: {torch.cuda.memory_allocated()} bytes")
print(f"Cached GPU memory: {torch.cuda.memory_reserved()} bytes")
