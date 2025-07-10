import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataset_data_synth(data_dir, batch_size):

    if 'mnist' in data_dir:
        data_mean = 0.5
        data_std = 0.5
        img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([data_mean], [data_std]),
        ])

        if "fashion" in data_dir:
            train_data = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,
                                                transform=img_transform)

            test_data = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True,
                                                transform=img_transform)
        else:
            train_data = torchvision.datasets.MNIST(
                root=data_dir, train=True, transform=img_transform, download=True)           
            test_data = torchvision.datasets.MNIST(
                root=data_dir, train=False, transform=img_transform, download=True)


        train_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=False)


    if 'cifar10' in data_dir:
        data_mean = (0.5, 0.5, 0.5)
        data_std = (0.5, 0.5, 0.5)
        img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
        ])

        cifar_train = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, transform=img_transform, download=True
        )
        train_loader = DataLoader(
            dataset=cifar_train, batch_size=batch_size, shuffle=True)

        cifar_test = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, transform=img_transform, download=True
        )
        test_loader = DataLoader(
            dataset=cifar_test, batch_size=batch_size, shuffle=False)

    val_loader = 0

    return train_loader, val_loader, test_loader, data_mean, data_std




