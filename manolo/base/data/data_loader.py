from manolo.base.data.cifar import get_dataset
from manolo.base.data.datasets_gan import get_dataset_data_synth

def load_dataset(args):
    """Loads the dataset using the provided arguments."""
    return get_dataset(args)


def load_dataset_data_synth(data_dir, batch_size):
    """Loads the dataset using the provided arguments."""
    return get_dataset_data_synth(data_dir, batch_size)
