from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN

def get_dataset(name, transform, root="./../data"):
    common_args = {
        "root": root,
        "transform": transform,
        "download": True
    }

    if name.upper() == "CIFAR10":
        return CIFAR10(train=True, **common_args)
    elif name.upper() == "CIFAR100":
        return CIFAR100(train=True, **common_args)
    elif name.upper() == "STL10":
        return STL10(split='train', **common_args)
    elif name.upper() == "SVHN":
        return SVHN(split='train', **common_args)
    else:
        raise ValueError(f"Unknown dataset: {name}")
