import torchvision.models as models
from torch import nn

def get_backbone(name, using_data=None):
    model = None
    name = str(name).lower()
    if name == "resnet18":
        model = models.resnet18()
    elif name == "resnet50":
        model = models.resnet50()
    else:
        raise ValueError(f"Unknown backbone model: {name}")
        exit(1)
    
    # CIFAR-10, CIFAR-100용 ResNet Stem 조정
    # 첫 번째 7x7 Conv of stride 2 -> 3x3 Conv of stride 1
    # 첫 번째 max pooling operation 제거
    if using_data.upper() == "CIFAR10" or using_data.upper() == "CIFAR100":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() 

    model = nn.Sequential(*list(model.children())[:-1])  # 마지막 FC Layer 제거

    return model