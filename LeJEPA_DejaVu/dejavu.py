import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping






def main():
    hyperparams = [
        backbone = 'resnet18', 'vit-s/8',
        model = 'simclr', 'barlowtwins', 'vicreg', 'lejepa'
        batch_size = 32
        learning_rate = 1e-3,
    ]

    