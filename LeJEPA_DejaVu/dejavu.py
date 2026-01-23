import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models import SimCLR, BarlowTwins, VICReg, LeJEPA
import datasets
from torchvision import transforms

from callback import OnlineLinearEvaluation
from sklearn.neighbors import NearestNeighbors
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset 하나를 통으로 representation으로 변환, Label 포함
def generate_whole_representation(model, dataloader):
    model.eval()
    representations = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            outputs = model(data)
            representations.append(outputs.cpu())
            labels.append(label.cpu())

    return torch.cat(representations, dim=0), torch.cat(labels, dim=0)

# kNN 탐색 및 예측 라벨 반환
def search_kNN(rep, whole, labels, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(whole.numpy())
    distances, indices = nbrs.kneighbors(rep.numpy())

    # 다수결 투표로 예측 라벨 결정
    pred_labels = []
    for idx_list in indices:
        neighbor_labels = labels[idx_list].numpy()
        counts = np.bincount(neighbor_labels)
        pred_label = np.argmax(counts)
        pred_labels.append(pred_label)

    return np.array(pred_labels)













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


    # crop the non-image image (background)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # todo; add the left-down corner crop for this dataset, indicating the background color
        # like in the original deja vu paper
        # as i don't know exact bounding box of the given stl10 dataset
    ])

    # datasets
    dataset_A = # 대충 1번재 모델 학습하기 좋은 거
    dataset_B = # 대충 2번재 모델 학습하기 좋은 거
    dataset_X = # 대충 비교용 representation을 위한 거 !!label 필요!!



    dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_X = torch.utils.data.DataLoader(dataset_X, batch_size=batch_size, shuffle=False, num_workers=4)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader_A):
            data = data.to(device)
            outputs1 = model1(data)

        for batch_idx, (data, _) in enumerate(dataloader_B):
            data = data.to(device)
            outputs2 = model2(data)
    
    # 적당히 이즘에서 callback

if __name__ == "__main__":
    main()