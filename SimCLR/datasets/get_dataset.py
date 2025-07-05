from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN
from torchvision import datasets

def get_dataset(name, transform, root):
    common_args = {
        "root": root,
        "transform": transform,
        "download": True
    }
    
    name = str(name).upper()

    if name == "CIFAR10":
        return CIFAR10(train=True, **common_args)
    elif name == "CIFAR100":
        return CIFAR100(train=True, **common_args)
    elif name == "STL10":
        return STL10(split='unlabeled', **common_args)
    elif name == "SVHN":
        return SVHN(split='train', **common_args)
    elif name == "DEEPFAKE":
        return datasets.ImageFolder(root=f"{root}/deepfake/train", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_test_dataset(name, transform, root):
    common_args = {
        "root": root,
        "transform": transform,
        "download": True
    }
    
    name = str(name).upper()

    if name == "CIFAR10":
        return CIFAR10(train=False, **common_args)
    elif name == "CIFAR100":
        return CIFAR100(train=False, **common_args)
    elif name == "STL10":
        return STL10(split='test', **common_args)
    elif name == "SVHN":
        return SVHN(split='test', **common_args)
    elif name == "DEEPFAKE":
        return datasets.ImageFolder(root=f"{root}/deepfake/test", transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")