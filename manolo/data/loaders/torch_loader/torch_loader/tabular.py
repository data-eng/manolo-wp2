import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from . import utils

logger = utils.get_logger(level='DEBUG')

def robust_normalize(dir, name, process, include, stats):
    """
    Normalize numpy dataset using robust scaling (median and IQR) from precomputed stats, include only specified columns, and save normalized data.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'test').
    :param include: List of column names to include in normalization.
    :param stats: Dict of precomputed stats (median, iqr) keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npy")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    column_names = metadata["columns"]
    data_norm = data.astype(np.float32).copy()

    for idx, col in enumerate(column_names):
        if col not in include:
            continue

        median = stats[col]['median']
        iqr = stats[col]['iqr'] if stats[col]['iqr'] > 0 else 1.0

        data_norm[:, idx] = (data[:, idx] - median) / iqr

    norm_path = utils.get_path(dir, filename=f"{name}-{process}-robust-norm.npy")
    utils.save_npy(data_norm, norm_path)

    logger.info(f"Robust normalized data saved to {norm_path}.")

def standard_normalize(dir, name, process, include, stats):
    """
    Normalize numpy dataset using standard scaling (mean and std) from precomputed stats, include only specified columns, and save normalized data.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'test').
    :param include: List of column names to include in normalization.
    :param stats: Dict of precomputed stats (mean, std) keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npy")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    column_names = metadata["columns"]
    data_norm = data.astype(np.float32).copy()

    for idx, col in enumerate(column_names):
        if col not in include:
            continue

        mean = stats[col]['mean']
        std = stats[col]['std'] if stats[col]['std'] > 0 else 1.0

        data_norm[:, idx] = (data[:, idx] - mean) / std

    norm_path = utils.get_path(dir, filename=f"{name}-{process}-std-norm.npy")
    utils.save_npy(data_norm, norm_path)

    logger.info(f"Normalized data saved to {norm_path}.")

def get_stats(dir, name):
    """
    Load numpy train data and metadata, compute stats (mean, std, median, IQR) per column, and save the stats as a JSON file.

    :param dir: Directory containing {name}-train.npy and {name}.json.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    :return: Dict of stats keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-train.npy")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    column_names = metadata["columns"]
    stats = {}

    for i, col in enumerate(column_names):
        col_values = data[:, i]

        mean = np.mean(col_values)
        std = np.std(col_values)
        median = np.median(col_values)
        q75, q25 = np.percentile(col_values, [75 ,25])
        iqr = q75 - q25

        stats[col] = {
            'mean': float(mean),
            'std': float(std),
            'median': float(median),
            'iqr': float(iqr)
        }

    stats_path = utils.get_path(dir, filename=f"{name}-stats.json")
    utils.save_json(data=stats, path=stats_path)

    logger.info(f"Saved statistics JSON to {stats_path}.")

    return stats

class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_epoch=True):
        """
        Initializes a time series dataset. It creates sequences from the input data by 
        concatenating features and time columns. The target variable is stored separately.

        :param df: Pandas dataframe containing the data.
        :param seq_len: Length of the input sequence (number of time steps).
        :param X: List of feature columns.
        :param t: List of time-related columns.
        :param y: List of target columns.
        :param per_epoch: Whether to create sequences in non-overlapping (True) or overlapping (False) epochs.
        """
        self.seq_len = seq_len
        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]
        self.per_epoch = per_epoch

        logger.debug(f'Initializing dataset with: samples={self.num_samples}, samples/seq={seq_len}, seqs={self.num_seqs}, epochs={self.num_epochs} ')

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        :return: Length of the dataset.
        """
        return self.num_seqs

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: Index of the sample.
        :return: Tuple of features and target tensors.
        """
        if self.per_epoch:
            start_idx = idx * self.seq_len
        else:
            start_idx = idx

        end_idx = start_idx + self.seq_len

        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values

        X, y = torch.FloatTensor(X), torch.LongTensor(y)

        return X, y
    
    @property
    def num_samples(self):
        """
        Returns the total number of samples in the dataset.
        
        :return: Total number of samples.
        """
        return self.X.shape[0]
    
    @property
    def num_epochs(self):
        """
        Returns the number of full epochs available based on the dataset size.

        :return: Number of epochs.
        """
        return self.num_samples // 7680

    @property
    def max_seq_id(self):
        """
        Returns the maximum index for a sequence.

        :return: Maximum index for a sequence.
        """
        return self.num_samples - self.seq_len
    
    @property
    def num_seqs(self):
        """
        Returns the number of sequences that can be created from the dataset.

        :return: Number of sequences.
        """
        if self.per_epoch:
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1
        
def create_dataset(df, seq_len, features, time, labels):
    """
    Create datasets for the specified dataframes (e.g. training, validation, and testing).

    :param df: Dataframe containing the data.
    :param seq_len: Sequence length for each dataset sample.
    :param features: List of feature columns.
    :param time: List of time-related columns.
    :param labels: List of label columns.
    :return: Dataset obejct.
    """
    logger.info('Creating dataset from dataframe.')     

    dataset = TSDataset(df=df, 
                        seq_len=seq_len, 
                        X=features,
                        t=time, 
                        y=labels)

    logger.debug(f'Dataset created successfully!')

    return dataset