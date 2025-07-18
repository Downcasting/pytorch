from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN, ImageFolder
from torchvision import datasets

def get_dataset(name, transform, root="data", pretrain=True):
    name = str(name).upper()
    
    if name == "CIFAR10":
        return CIFAR10(root=root, train=True, transform=transform, download=True)

    elif name == "CIFAR100":
        return CIFAR100(root=root, train=True, transform=transform, download=True)

    elif name == "STL10":
        split = 'unlabeled' if pretrain else 'train'
        return STL10(root=root, split=split, transform=transform, download=True)

    elif name == "SVHN":
        return SVHN(root=root, split='train', transform=transform, download=True)

    elif name == "DEEPFAKE":
        subfolder = "train" if pretrain else "finetune"
        return ImageFolder(root=f"{root}/deepfake/{subfolder}", transform=transform)

    else:
        raise ValueError(f"[get_dataset] Unknown dataset: {name}")

def get_test_dataset(name, transform, root="data"):
    name = str(name).upper()

    if name == "CIFAR10":
        return CIFAR10(root=root, train=False, transform=transform, download=True)

    elif name == "CIFAR100":
        return CIFAR100(root=root, train=False, transform=transform, download=True)

    elif name == "STL10":
        return STL10(root=root, split='test', transform=transform, download=True)

    elif name == "SVHN":
        return SVHN(root=root, split='test', transform=transform, download=True)

    elif name == "DEEPFAKE":
        return ImageFolder(root=f"{root}/deepfake/test", transform=transform)

    else:
        raise ValueError(f"[get_test_dataset] Unknown dataset: {name}")
