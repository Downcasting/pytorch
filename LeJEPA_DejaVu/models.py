

import torch
import torch.nn as nn
import pytorch_lightning as pl

# 여기에 그냥 넣지 말고, 모델별 loss function까지 구현한 module 넣어놓기

class SimCLR(nn.Module):
    def __init__(self, base_encoder: nn.Module, projection_dim: int = 128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder


    def forward(self, x):
        features = self.encoder(x)
        # projections = self.projection_head(features)
        return features
    

class BarlowTwins(nn.Module):
    def __init__(self, base_encoder: nn.Module, projection_dim: int = 128):
        super(BarlowTwins, self).__init__()
        self.encoder = base_encoder
        
    def forward(self, x):
        features = self.encoder(x)
        return features
    
class VICReg(nn.Module):
    def __init__(self, base_encoder: nn.Module, projection_dim: int = 128):
        super(VICReg, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.output_dim, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections
    
class LeJEPA(nn.Module):
    def __init__(self, base_encoder: nn.Module, projection_dim: int = 128):
        super(LeJEPA, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.output_dim, projection_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return projections
    
