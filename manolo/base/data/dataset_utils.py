import torch 
import torchvision.transforms as transforms
import torchvision.datasets as dst
import pickle as pkl
from torch.utils.data import Dataset

from IPython import embed
import numpy as np


class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path):
        with open(features_path, 'rb') as f:
            self.features = pkl.load(f)
        with open(labels_path, 'rb') as f:
            self.targets = pkl.load(f)

        # embed()
       
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.targets[idx]
        return feature, label


def get_features_dataset(args):

    print("Loading training featrues...")
    train_set = FeatureDataset(args.train_feat_file, args.train_lab_file)
    print("Loading testing featrues...")
    test_set = FeatureDataset(args.test_feat_file, args.test_lab_file)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.train_shuffle, num_workers=args.n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    return train_loader, test_loader

def get_stats(data_name, pretrained_model):

    # define transforms
    if data_name == 'cifar10':
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010) 
    elif data_name == 'cifar100':
        dataset = dst.CIFAR100
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010) 
    else:
        raise Exception('Invalid dataset name...')
    

    if pretrained_model=="True": # load stats from ImageNet pretraining.
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)         
    
    return mean, std, dataset



def get_dataset(args):

    mean, std, dataset = get_stats(args.data_name, args.pretrained_model)

    test_transform = 0

    # test_transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      #transforms.Resize((225, 225)),
    #      transforms.Normalize(mean, std)])

    if args.data_name == "cifar100":    
        # Transformations from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py
        train_transform = 0
        
    elif args.data_name == "cifar10":        
        # Transformations from https://github.com/kuangliu/pytorch-cifar/tree/master
        train_transform = 0   
    else:
        print("Dataset name not found...")
        exit()

    # define data loader
    if args.transform_train==False:
        train_transform = test_transform

    train_loader = torch.utils.data.DataLoader(
            dataset(root      = args.img_root,
                    transform = train_transform,
                    train     = True,
                    download  = True),
            batch_size=args.batch_size, shuffle=args.train_shuffle, num_workers=args.n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            dataset(root      = args.img_root,
                    transform = test_transform,
                    train     = False,
                    download  = True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)


    return train_loader, test_loader


def preproces_and_normalise(X_train, X_val, norm_type="min_max", round_two_decimals=True):
    X_train_r = X_train
    X_val_r = X_val
    
    if norm_type=="min_max":
        max_value = X_train_r.max()
        min_value = X_train_r.min()

        # min-max norm
        X_train_r = (X_train_r - min_value)/(max_value - min_value)
        X_val_r = (X_val_r - min_value)/(max_value - min_value)

    elif norm_type=="between_m1_and_1":
        mean_val = X_train_r.mean()

        # Center the data around cero 
        X_train_r = X_train_r - mean_val
        X_val_r = X_val_r - mean_val

        min_val = X_train_r.min()
        max_val = X_train_r.max()

        # Scale data between -1 and 1
        scale_value = max(abs(min_val), abs(max_val))
        X_train_r = X_train_r/scale_value
        X_val_r = X_val_r/scale_value


    if round_two_decimals:
        # Round to two decimals
        X_train = np.round(X_train_r, 2)
        X_val = np.round(X_val_r, 2)

    return X_train, X_val