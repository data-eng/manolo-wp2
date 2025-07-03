# import torch
# import random
# import multiprocessing
# import pandas as pd
# import numpy as np
# import dask.dataframe as dd
# from torch.utils.data import Dataset, DataLoader
# import os
# import io
# import s3fs
# import json
# import mne

# from . import utils

# logger = utils.get_logger(level='DEBUG')

# class TSDataset(Dataset):
#     def __init__(self, df, seq_len, X, t, y, per_epoch=True):
#         """
#         Initializes a time series dataset. It creates sequences from the input data by 
#         concatenating features and time columns. The target variable is stored separately.

#         :param df: Pandas dataframe containing the data.
#         :param seq_len: Length of the input sequence (number of time steps).
#         :param X: List of feature columns.
#         :param t: List of time-related columns.
#         :param y: List of target columns.
#         :param per_epoch: Whether to create sequences in non-overlapping (True) or overlapping (False) epochs.
#         """
#         self.seq_len = seq_len
#         self.X = pd.concat([df[X], df[t]], axis=1)
#         self.y = df[y]
#         self.per_epoch = per_epoch

#         logger.debug(f'Initializing dataset with: samples={self.num_samples}, samples/seq={seq_len}, seqs={self.num_seqs}, epochs={self.num_epochs} ')

#     def __len__(self):
#         """
#         Returns the number of sequences in the dataset.

#         :return: Length of the dataset.
#         """
#         return self.num_seqs

#     def __getitem__(self, idx):
#         """
#         Retrieves a sample from the dataset at the specified index.

#         :param idx: Index of the sample.
#         :return: Tuple of features and target tensors.
#         """
#         if self.per_epoch:
#             start_idx = idx * self.seq_len
#         else:
#             start_idx = idx

#         end_idx = start_idx + self.seq_len

#         X = self.X.iloc[start_idx:end_idx].values
#         y = self.y.iloc[start_idx:end_idx].values

#         X, y = torch.FloatTensor(X), torch.LongTensor(y)

#         return X, y
    
#     @property
#     def num_samples(self):
#         """
#         Returns the total number of samples in the dataset.
        
#         :return: Total number of samples.
#         """
#         return self.X.shape[0]
    
#     @property
#     def num_epochs(self):
#         """
#         Returns the number of full epochs available based on the dataset size.

#         :return: Number of epochs.
#         """
#         return self.num_samples // 7680

#     @property
#     def max_seq_id(self):
#         """
#         Returns the maximum index for a sequence.

#         :return: Maximum index for a sequence.
#         """
#         return self.num_samples - self.seq_len
    
#     @property
#     def num_seqs(self):
#         """
#         Returns the number of sequences that can be created from the dataset.

#         :return: Number of sequences.
#         """
#         if self.per_epoch:
#             return self.num_samples // self.seq_len
#         else:
#             return self.max_seq_id + 1

# def preprocess(dir, name, exist=False):
#     """
#     Preprocess the data by extracting weights and normalizing the features.

#     :param dir: Directory to save the processed data.
#     :param name: Name of the dataset (e.g., 'bitbrain-train').
#     :param exist: Boolean flag indicating if the processed file already exists.
#     :return: Processed NumPy array and weights dictionary.
#     """
#     stats = None
#     ds = name.split('-')[0]
#     process = name.split('-')[1].split('.')[0]

#     if exist:
#         path = utils.get_path(dir, filename=f'{name}-norm.npy')
#         data = utils.load_npy(path)
#         logger.info(f'Loaded existing data from {path}.')

#         weights_path = utils.get_path(dir, filename=f'{ds}-weights.json')
#         weights = utils.load_json(weights_path)
#         logger.info(f'Loaded existing weights from {weights_path}. These are: {weights}.')

#         return data, weights
    
#     # do label mapping for each label in cols called labels (in dir there is metadata dont forget that)
    
#     if process == 'train':
#         stats_path = utils.get_path(dir, filename='stats.json')
#         stats = utils.get_stats(df=df[original_cols])

#         utils.save_json(data=stats, filename=stats_path)
#         logger.info(f'Saved stats to {stats_path} from {name} dataframe.')

#         weights = extract_weights(df=df, label_col='Label', dir=dir)
#         logger.info(f'Extracted weights from {name} dataframe. These are: {weights}.')
#     else:
#         stats_path = utils.get_path(dir, filename='stats.json')
#         stats = utils.load_json(stats_path)

#         weights_path = utils.get_path(dir, filename='weights.json')
#         weights = utils.load_json(weights_path)
#         logger.info(f'Loaded weights from {weights_path}. These are: {weights}.')

#     df = utils.standard_normalize(df=df, exclude=exclude_cols, stats=stats)

#     path = utils.get_path(dir, filename=f'{name}.csv')
#     df.to_csv(path, index=False)

#     logger.info(f'Saved {name} dataframe to {path} with shape: {df.shape}. First few rows:\n{df.head()}.') 

#     return df, weights

# def create_dataset(df, seq_len, features, time, labels):
#     """
#     Create datasets for the specified dataframes (e.g. training, validation, and testing).

#     :param df: Dataframe containing the data.
#     :param seq_len: Sequence length for each dataset sample.
#     :param features: List of feature columns.
#     :param time: List of time-related columns.
#     :param labels: List of label columns.
#     :return: Dataset obejct.
#     """
#     logger.info('Creating dataset from dataframe.')     

#     dataset = TSDataset(df=df, 
#                         seq_len=seq_len, 
#                         X=features,
#                         t=time, 
#                         y=labels)

#     logger.debug(f'Dataset created successfully!')

#     return dataset