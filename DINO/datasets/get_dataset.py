from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN, ImageFolder
from torchvision import datasets
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from transform import DINODualTransform

class DINODataModule(LightningDataModule):
    def __init__(self, name, transform, root="./../data", batch_size=128, num_workers=4, cfg=None):
        super().__init__()
        self.name = str(name).upper()
        self.transform = DINODualTransform(cfg)
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

    def prepare_data(self):
        if self.name == "CIFAR10":
            CIFAR10(root=self.root, train=True, transform=self.transform, download=True)
            CIFAR10(root=self.root, train=False, transform=self.transform, download=True)
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

    def setup(self, stage=None):
        if self.name == "CIFAR10":
            self.train_dataset = CIFAR10(root=self.root, train=True, transform=self.transform, download=False)
            self.val_dataset = CIFAR10(root=self.root, train=False, transform=self.transform, download=False)
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)



