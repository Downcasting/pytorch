import torch
import torch.nn as nn

# 1. Lazy Method
# 단점: 특정 class에 종속됨, class 정의가 바뀌면 저장된 모델을 불러올 수 없음, 보안 문제 있음
torch.save(arg, PATH)
model = torch.load(PATH)
model.eval()

# 2. Recommended Method
# Inference를 위한 모델을 저장할 때는 이 방법을 사용
# 가중치 '만' 저장한 거여서, 나중에 torch.load(PATH) 뒷부분에 weights_only=False를 안 넣어줘도 성립함
# 그 대신, 완전 다른 모델에 이상한 가중치를 불러올 수 있음
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()