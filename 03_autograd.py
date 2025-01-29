import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

z = z.mean()
z.backward() # dz/dx
print('z is ')
print(x.grad)

# Gradient 계산 중단
# x.requires_grad_(False)
# x.detach() -> new tensor with same value but no gradient
# with torch.no_grad(): -> code block without gradient 

x.requires_grad_(False)
print(x)
y.detach_()
print(y)

with torch.no_grad():
    z = z + 2
    print(z)


# weight update
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

weights.grad.zero_()

# Stochastic Gradient Descent

weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD([weights], lr=0.01)
optimizer.step()
optimizer.zero_grad()