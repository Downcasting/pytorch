import torch
import torch.nn as nn
import numpy as np

'''
Cross Entropy Loss
CE(y, y_hat) = - sum(y_i * log(y_hat_i))
'''

'''
Y has class labels, not one-hot
Y_pred has raw scores (logits), no softmax
'''
loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1])

# nsamples x nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [3.1, 1.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

_, predictions1 = torch.max(Y_pred_good, 1) 
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)

'''
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

# y_hat are the predicted probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')
'''