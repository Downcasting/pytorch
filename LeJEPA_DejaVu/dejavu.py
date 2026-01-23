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

from sklearn.model_selection import train_test_split
import wandb
from pytorch_lightning.loggers import WandbLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
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
'''

# kNN 탐색 및 예측 라벨 반환
def search_kNN(rep, whole, labels, k=100):
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

    wandb_logger = WandbLogger(project="colored-mnist")

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

    # 메인 dataset, 나중에 꼭 다른 것으로 바꾸기
    # 아마 높은 확률로 imagenet-100 mini를 쓸 것 같은데
    dataset = datasets.STL10_DejaVu(root='./data', split='unlabeled', transform=transform, download=True)

    dataset_A, dataset_BX = train_test_split(dataset, test_size=0.6, random_state=42)
    dataset_B, dataset_X = train_test_split(dataset_BX, test_size=0.33, random_state=42)

    dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_X = torch.utils.data.DataLoader(dataset_X, batch_size=batch_size, shuffle=False, num_workers=4)


    ##############################################################
    ############# 여기부터 모델 Pretraining하는 단계 ################
    ##############################################################

    online_eval_callback = OnlineLinearEvaluation(
        num_classes=100,
        feature_dim=,      # output dim 설정
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[online_eval_callback]
    )


    ##############################################################
    ############# 여기부터 실제로 DejaVu Score 계산 #################
    ##############################################################

    model1.eval()
    model2.eval()

    A_acc = 0
    B_acc = 0

    # 실제로 representation kNN을 이용한 accuracy 계산
    with torch.no_grad():
        A_rep, A_labels = generate_whole_representation(model1, dataloader_X)
        B_rep, B_labels = generate_whole_representation(model2, dataloader_X)

        for _, (data, labels) in enumerate(dataloader_A):
            data = data.to(device)
            outputs_a = model1(data)

            answer_a = search_kNN(outputs_a.cpu(), A_rep, A_labels, k=100)
            A_acc += np.sum(answer_a == labels.numpy())

        for _, (data, labels) in enumerate(dataloader_B):
            data = data.to(device)
            outputs_b = model2(data)

            answer_b = search_kNN(outputs_b.cpu(), B_rep, B_labels, k=100)
            B_acc += np.sum(answer_b == labels.numpy())

        A_acc = A_acc / len(dataset_A)
        B_acc = B_acc / len(dataset_B)

    dejavu_score = A_acc - B_acc
    print(f"DejaVu Score: {dejavu_score}, Model 1 Acc: {A_acc}, Model 2 Acc: {B_acc}")

if __name__ == "__main__":
    main()