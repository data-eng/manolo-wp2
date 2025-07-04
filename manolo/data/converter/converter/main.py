import json
import numpy as np

def create_metadata(path, columns):
    """
    Save metadata for structured numpy arrays into a JSON file.

    :param path: Path to save the metadata JSON.
    :param columns: Tuple containing (features, time, labels, split, weights, other).
    """
    features, time, labels, split, weights, other = columns

    metadata = {
        "columns": features + time + labels + other,
        "features": features,
        "time": time,
        "labels": labels,
        "split": split,
        "weights": weights,
        "other": other
    }

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def create_npz(data, path):
    """
    Save multiple structured numpy arrays into a single .npz file.

    :param data: Tuple of arrays (features, time, labels, split, weights, other).
    :param path: Path where the .npz archive will be saved.
    """
    keys = ['features', 'time', 'labels', 'split', 'weights', 'other']
    array_dict = {k: v for k, v in zip(keys, data)}

    np.savez(path, **array_dict)