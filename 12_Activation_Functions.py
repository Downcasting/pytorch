import torch
import torch.nn as nn
import torch.nn.functional as F

# option 1 (create nn modules)
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        '''
        nn.Sigmoid() applies the sigmoid function element-wise
        nn.Softmax(dim=1) applies the softmax function to
        nn.TanH() applies the hyperbolic tangent function element-wise
        nn.LeakyReLU() applies the leaky relu function element-wise
        '''
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    
# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        '''
        torch.sigmoid() applies the sigmoid function element-wise
        torch.softmax(dim=1) applies the softmax function to
        torch.tanh() applies the hyperbolic tangent function element-wise
        torch.leaky_relu() applies the leaky relu function element-wise
        '''