import time
import pickle as pkl
import torch
from tqdm import tqdm
from ..base.data.data_loader import load_dataset
from ..base.models.network_initializer import initialize_network
from ..base.metrics.parameter_counter import count_parameters_in_MB
import numpy as np
import torch.backends.cudnn as cudnn


def extract_features(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
        mps_device = None
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        device = mps_device
    else:
        mps_device = None
        device = torch.device("cpu")

    print("args = %s", args)
    #print("unparsed_args = %s", unparsed)


    """Main function to extract features."""
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset(args)

    net, feat_dim = initialize_network(args, device)
    print("Network initialized with feature dimension:", feat_dim)
    print("Parameter size (MB):", count_parameters_in_MB(net))
    print('Using device:', device)

    net.eval()
    with torch.no_grad():
        # Extract training features
        feats, labs = [], []
        for img, target in tqdm(train_loader, desc="Training Features"):
            img, target = img.to(device), target.to(device)
            feats.extend(net(img).cpu().numpy())
            labs.extend(target.cpu().numpy())
        with open(args.train_feat_file, 'wb') as f:
            pkl.dump(feats, f)
        with open(args.train_lab_file, 'wb') as f:
            pkl.dump(labs, f)

        # Extract testing features
        feats, labs = [], []
        for img, target in tqdm(test_loader, desc="Testing Features"):
            img, target = img.to(device), target.to(device)
            feats.extend(net(img).cpu().numpy())
            labs.extend(target.cpu().numpy())
        with open(args.test_feat_file, 'wb') as f:
            pkl.dump(feats, f)
        with open(args.test_lab_file, 'wb') as f:
            pkl.dump(labs, f)
