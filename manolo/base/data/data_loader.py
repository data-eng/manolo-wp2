from manolo.base.data.cifar import get_dataset

def load_dataset(args):
    """Loads the dataset using the provided arguments."""
    return get_dataset(args)
