from ..utils.feat_utils import initilise_architecture

def initialize_network(args, device):
    """Initializes the network and returns the model and feature dimensions."""
    return initilise_architecture(args, device)

