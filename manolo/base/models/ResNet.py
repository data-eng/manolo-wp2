import torch
import torchvision
from torchvision.models import (
    resnet18, ResNet18_Weights
)

def load_resnet18(pretrained=True):
    # Load pretrained ResNet-18 model
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    #model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)

    print(' --- ResNet Loaded --- ')
    return model