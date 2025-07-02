import json
import numpy as np

def create_metadata(path, column_names, features, time, labels, other):
    """
    Create a metadata JSON file with the specified information and save it to the given path.

    :param path: Path where the metadata JSON file will be saved.
    :param column_names: List of column names in the DataFrame.
    :param features: List of feature names.
    :param time: List of time-related column names.
    :param labels: List of label names.
    :param other: List of other relevant information.
    """
    metadata = {
        "columns": column_names,
        "features": features,
        "time": time,
        "labels": labels,
        "other": other
    }

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def create_npy(df, path):
    """
    Convert a pandas DataFrame to a numpy array and save it to the specified path.

    :param df: DataFrame to convert.
    :param path: Path where the numpy array will be saved.
    """
    np.save(path, df.values.astype(np.float32))