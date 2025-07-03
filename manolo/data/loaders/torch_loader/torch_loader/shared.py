import random
import multiprocessing
import numpy as np
import os
from torch.utils.data import DataLoader

from . import utils

logger = utils.get_logger(level='DEBUG')

def split_data(dir, name, train_size=0.75, val_size=0.25, test_size=0, exist=False):
    """
    Split the data into training, validation, and testing sets based on unique night_id values.

    :param dir: Directory containing the numpy array.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :param train_size: Proportion of nights to use for training.
    :param val_size: Proportion of nights to use for validation.
    :param test_size: Proportion of nights to use for testing.
    :param exist: Boolean flag indicating if the training, validation, and testing dataframes already exist.
    :return: Tuple of DataFrames for training, validation, and testing.
    """
    data_path = utils.get_path(dir, filename=f'{name}.npy')
    meta_path = utils.get_path(dir, filename=f'{name}.json')

    train_path = utils.get_path(dir, filename=f'{name}-train.npy')
    val_path = utils.get_path(dir, filename=f'{name}-val.npy')
    test_path = utils.get_path(dir, filename=f'{name}-test.npy')

    if exist:
        train_data = utils.load_npy(train_path)
        val_data = utils.load_npy(val_path)
        test_data = utils.load_npy(test_path)

        logger.info(f"Loaded existing dataframes from {dir}.")

        return train_data, val_data, test_data
    else:
        data = utils.load_npy(data_path)
        metadata = utils.load_json(meta_path)

        logger.info(f"Loaded data from {data_path} and metadata from {meta_path}.")

        column_names = metadata["columns"]
        split_col = metadata["split"]

        if split_col is None:
            total = len(data)
            indices = list(range(total))

            random.seed(42)
            random.shuffle(indices)

            train_end = int(total * train_size)
            val_end = train_end + int(total * val_size)

            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]

            train_data = data[train_idx]
            val_data = data[val_idx]
            test_data = data[test_idx]
        
        else:
            split_col_idx = column_names.index(split_col)
            unique_values = list(np.unique(data[:, split_col_idx]))

            logger.info(f"Unique values for split column '{split_col}': {unique_values}.")

            random.seed(42)
            random.shuffle(unique_values)

            n = len(unique_values)

            if train_size + val_size + test_size == 0:
                raise ValueError("All sets have size 0, which is invalid.")
            if train_size + val_size + test_size > 1:
                raise ValueError("Sum of train, val, and test sizes must not exceed 1.")
            
            raw = {'train': round(n * train_size), 'val': round(n * val_size), 'test': round(n * test_size)}
            total = sum(raw.values())

            while total > n:
                max_key = max(raw, key=raw.get)
                raw[max_key] -= 1
                total -= 1
                
            while total < n:
                min_key = min(raw, key=raw.get)
                raw[min_key] += 1
                total += 1
           
            train_end = raw['train']
            val_end = train_end + raw['val']

            train_vals = set(unique_values[:train_end])
            val_vals = set(unique_values[train_end:val_end])
            test_vals = set(unique_values[val_end:])

            def filter(values):
                values = np.array(list(values)).astype(data[:, split_col_idx].dtype)
                return data[np.isin(data[:, split_col_idx], values)].copy()
            
            train_data = filter(train_vals)
            val_data = filter(val_vals)
            test_data = filter(test_vals)

            logger.info(f"Train values: {sorted(train_vals)}")
            logger.info(f"Validation values: {sorted(val_vals)}")
            logger.info(f"Test values: {sorted(test_vals)}")

            assert train_vals.isdisjoint(val_vals), "Overlap in train and val nights!"
            assert train_vals.isdisjoint(test_vals), "Overlap in train and test nights!"
            assert val_vals.isdisjoint(test_vals), "Overlap in val and test nights!"   

        utils.save_npy(data=train_data, path=train_path)
        utils.save_npy(data=val_data, path=val_path)
        utils.save_npy(data=test_data, path=test_path)

        logger.info(f"Data split into train ({len(train_data)} samples), val ({len(val_data)} samples), test ({len(test_data)} samples).")

        return train_data, val_data, test_data

def extract_weights(dir, name):
    """
    Calculate class weights from the training NumPy array to handle class imbalance, and save them to a file.

    :param dir: Directory to save the weights file.
    :param name: Name of the dataset (e.g., 'bitbrain').
    :return: Dictionary of class weights.
    """
    data_path = utils.get_path(dir, filename=f'{name}-train.npy')
    meta_path = utils.get_path(dir, filename=f'{name}.json')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}. Cannot extract weights.")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    column_names = metadata["columns"]
    weights_col = metadata["weights"]

    col_idx = column_names.index(weights_col)
    labels = data[:, col_idx]
    unique_labels, counts = np.unique(labels, return_counts=True)

    occs = dict(zip(unique_labels, counts))
    inverse_occs = {key: 1 / (value + 1e-10) for key, value in occs.items()}

    total = sum(inverse_occs.values())
    weights = {int(key): val / total for key, val in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    weights_path = utils.get_path(dir, filename=f'{name}-weights.json')
    utils.save_json(data=weights, path=weights_path)
    logger.info(f"Saved class weights to {weights_path}: {weights}")

    return weights

def shift_labels(dir, name):
    """
    Load the dataset and metadata, shift label columns' values to start from 0,
    and save the updated numpy data back to the same path.

    :param dir: Directory containing the dataset and metadata files.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    """
    data_path = utils.get_path(dir, filename=f"{name}.npy")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    column_names = metadata["columns"]
    label_cols = metadata["labels"]

    for col in label_cols:
        col_idx = column_names.index(col)
        col_values = data[:, col_idx]

        unique_vals = np.unique(col_values)
        mapping = {val: i for i, val in enumerate(sorted(unique_vals))}

        mapping_function = np.vectorize(mapping.get)
        remapped_values = mapping_function(col_values)

        data[:, col_idx] = remapped_values

    utils.save_npy(data=data, path=data_path)
    logger.info(f"Shifted labels {label_cols} in {data_path} so values start at 0.")

def create_dataloader(ds, batch_size, shuffle=[True, False, False], num_workers=None, drop_last=False):
    """
    Create DataLoader object for the specified dataset.

    :param ds: Dataset object.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data at every epoch.
    :param num_workers: Number of subprocesses to use for data loading (default is all available CPU cores).
    :param drop_last: Whether to drop the last incomplete batch.
    :return: DataLoader object.
    """
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    logger.info(f'System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.')

    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dl