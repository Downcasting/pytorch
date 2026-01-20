import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models import SimCLR, BarlowTwins, VICReg, LeJEPA
import datasets
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    ####### HYPERPARAMS #######

    batch_size = 32
    learning_rate = 1e-3
    max_epochs = 1000
    using_model = "LeJEPA" # 선택지: "SimCLR", "BarlowTwins", "VICReg", "LeJEPA"
    
    ###########################

    basic_parameters = {
        "base_encoder": pl.models.resnet.ResNet18(),
        "projection_dim": 128
    }

    if using_model == "SimCLR":
        model1 = SimCLR(basic_parameters)
        model2 = SimCLR(basic_parameters)
    elif using_model == "BarlowTwins":
        model1 = BarlowTwins(basic_parameters)
        model2 = BarlowTwins(basic_parameters)
    elif using_model == "VICReg":
        model1 = VICReg(basic_parameters)
        model2 = VICReg(basic_parameters)
    elif using_model == "LeJEPA":
        model1 = LeJEPA(basic_parameters)
        model2 = LeJEPA(basic_parameters)
    else:
        raise ValueError("Invalid model name. Choose from 'SimCLR', 'BarlowTwins', 'VICReg', 'LeJEPA'.")


    model1.load_state_dict(torch.load('model1.pth'))
    model1.to(device)
    model2.load_state_dict(torch.load('model2.pth'))
    model2.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # todo; add the left-down corner crop for this dataset, indicating the background color
        # like in the original deja vu paper
        # as i don't know exact bounding box of the given stl10 dataset
    ])

    dataset = datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)