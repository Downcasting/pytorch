import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models import SimCLR, BarlowTwins, VICReg, LeJEPA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    ####### HYPERPARAMS #######

    batch_size = 32
    learning_rate = 1e-3
    max_epochs = 1000
    using_model = "LeJEPA" # 선택지: "SimCLR", "BarlowTwins", "VICReg", "LeJEPA"
    
    ###########################

    parameters = {
        "base_encoder": pl.models.resnet.ResNet18(),
        "projection_dim": 128
    }

    if using_model == "SimCLR":
        model1 = SimCLR(parameters)
        model2 = SimCLR(parameters)
    elif using_model == "BarlowTwins":
        model1 = BarlowTwins(parameters)
        model2 = BarlowTwins(parameters)
    elif using_model == "VICReg":
        model1 = VICReg(parameters)
        model2 = VICReg(parameters)
    elif using_model == "LeJEPA":
        model1 = LeJEPA(parameters)
        model2 = LeJEPA(parameters)
    else:
        raise ValueError("Invalid model name. Choose from 'SimCLR', 'BarlowTwins', 'VICReg', 'LeJEPA'.")


    model1.load_state_dict(torch.load('model1.pth'))
    model1.to(device)
    model2.load_state_dict(torch.load('model2.pth'))
    model2.to(device)

