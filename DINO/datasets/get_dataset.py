from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN, ImageFolder
from torchvision import datasets
from torch.utils.data import DataLoader

def DINODataModule(name, transform, root="data", batch_size=128, num_workers=4):
    name = str(name).upper()
    dataset = None
    if name == "CIFAR10":
        dataset = CIFAR10(root=root, train=True, transform=transform, download=True)

    else:
        raise ValueError(f"[get_dataset] Unknown dataset: {name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def DINOTestDataModule(name, transform, root="data"):
    name = str(name).upper()

    if name == "CIFAR10":
        dataset = CIFAR10(root=root, train=False, transform=transform, download=True)

    else:
        raise ValueError(f"[get_test_dataset] Unknown dataset: {name}")
    

