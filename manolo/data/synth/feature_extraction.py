from manolo.base.wrappers.other_packages import pickle as pkl
from manolo.base.wrappers.other_packages import tqdm
from manolo.base.wrappers.numpy import np
from manolo.base.wrappers.pytorch import cudnn, torch

from manolo.base.data.data_loader import load_dataset
from manolo.data.synth.feature_extraction_utils import initilise_architecture
from manolo.base.utils.evaluation_utils import count_parameters_in_MB

from manolo.base.metrics.code_carbon_utils import codecarbon_manolo
from manolo.base.utils.logger_utils import log_data

@codecarbon_manolo
def extract_features(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.select_cuda != -1:
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

    """Main function to extract features."""
    train_loader, test_loader = load_dataset(args)

    net, feat_dim = initilise_architecture(args, device)
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


    print("Features stored locally.")
    if args.store_features_in_mlflow:
        log_data(args.train_feat_file, type="path")
        log_data(args.train_lab_file, type="path")
        log_data(args.test_feat_file, type="path")
        log_data(args.test_lab_file, type="path")
        print("Features stored in MLFlow.")
