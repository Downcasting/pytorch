import torch
import numpy as np

# 다양한 Tensor 생성 방법
x = torch.rand(2,3,4)
y = torch.zeros(2,2)
z = torch.ones(2,2)
p = torch.rand(2,2, dtype=torch.float64)
q = torch.tensor([2.0, 3.0])

# Tensor 연산
y = torch.rand(2,2)
z = torch.rand(2,2)

x1 = y + z
x2 = torch.add(y, z)
x3 = y.add(z)
y.add_(z)

y1 = y - z
y2 = torch.sub(y, z)
y3 = y.sub(z)
y.sub_(z)

z1 = y * z
z2 = torch.mul(y, z)
z3 = y.mul(z)
y.mul_(z)


# Tensor 일부 열만 가져오기
x = torch.rand(2,3)
print(x[:, 0])

# Tensor 크기 변경
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)

# Tensor가 CPU에서 사용된다면 (y는 x를 따라감)
x = torch.ones(2,2)
y = x.numpy()
x.add_(1)

x = np.ones((2,2))
y = torch.from_numpy(x)
x = np.add(x, 1)

# Tensor를 GPU로 사용하기
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones(2,2, device=device)
    x = torch.ones(2,2).to(device)
    z = x + y
    z = z.to("cpu", torch.double)
    print(z)
    z=z.numpy()
    print(z)

# Gradient 계산
x = torch.ones(2,2, requires_grad=True)