import torch
import torch.nn as nn

# GPU -> CPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))



# GPU -> GPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)


# CPU -> GPU
torch.save(model.state_dict(), PATH)

device = torch.device('cuda')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0")) # Choose whatever GPU device number you want
model.to(device)