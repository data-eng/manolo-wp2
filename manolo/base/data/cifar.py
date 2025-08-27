import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dst
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



def get_stats(args):

    # define transforms
    if args.data_name == 'cifar10':
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010) 
    elif args.data_name == 'cifar100':
        dataset = dst.CIFAR100
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010) 
    else:
        raise Exception('Invalid dataset name...')
    
    if args.pretrained_model==True: # load stats from ImageNet pretraining.
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)         
    
    return mean, std, dataset



def get_dataset(args):

    mean, std, dataset = get_stats(args)

    image_size = args.image_size

    test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])

    if args.data_name == "cifar100":    
        # Transformations from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.data_name == "cifar10":        
        # Transformations from https://github.com/kuangliu/pytorch-cifar/tree/master
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])    
    else:
        raise Exception('Dataset loader not implemented...')

    # define data loader
    if args.transform_train==False:
        train_transform = test_transform

    train_loader = DataLoader(
            dataset(root      = args.img_root,
                    transform = train_transform,
                    train     = True,
                    download  = True),
            batch_size=args.batch_size, shuffle=args.train_shuffle, num_workers=args.n_workers, pin_memory=True)
    test_loader = DataLoader(
            dataset(root      = args.img_root,
                    transform = test_transform,
                    train     = False,
                    download  = True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)


    return train_loader, test_loader


