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

print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)
print("cuDNN 버전:", torch.backends.cudnn.version())
print("GPU 개수:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))