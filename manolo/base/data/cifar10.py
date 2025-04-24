import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])


    train_dataset = torchvision.datasets.CIFAR10(root="./manolo/base/data/dataset", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./manolo/base/data/dataset", train=False, download=True, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(' --- CIFAR10 Data Loaded ---')

    return train_loader, test_loader