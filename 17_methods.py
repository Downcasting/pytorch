import torch
import torch.nn as nn

# 어떤 dictionary든 저장할 수 있음
torch.save(arg, PATH)

torch.load(PATH)

model.load_state_dict(arg)

# 심지어 optimizer의 가중치(Learning Rate 등...)도 저장할 수 있음
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())




# Save
model = Model(n_input_features=6)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Load
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)