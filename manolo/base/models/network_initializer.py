from ..utils.feat_utils import initilise_architecture
from ..utils.data_synth_utils import initilise_data_synth_model

def initialize_network(args, device):
    """Initializes the network and returns the model and feature dimensions."""
    return initilise_architecture(args, device)


def initialize_data_synth(args, device):
    """Initializes the gan and returns the model."""
    return initilise_data_synth_model(args, device)
