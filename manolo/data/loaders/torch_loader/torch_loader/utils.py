import json
import logging
import numpy as np
import os

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

def get_dir(*sub_dirs):
    """
    Retrieve or create a directory path based on the script's location and the specified subdirectories.

    :param sub_dirs: List of subdirectories to append to the script's directory.
    :return: Full path to the directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def get_path(*dirs, filename):
    """
    Construct a full file path by combining directory paths and a filename.

    :param dirs: List of directory paths.
    :param filename: Name of the file.
    :return: Full path to the file.
    """
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path

def save_npy(data, path):
    """
    Save a numpy array to a .npy file in the specified directory.

    :param data: Numpy array to save.
    :param path: Full path to the .npy file.
    """
    np.save(path, data)

def load_npy(path):
    """
    Load a .npy file from the given path.

    :param path: Full path to the .npy file.
    :return: Loaded numpy array.
    """
    return np.load(path)

def load_json(path):
    """
    Load a JSON file from the given path.

    :param path: Full path to the .json file.
    :return: Parsed JSON as a Python dict.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    """
    Save a Python dict to a JSON file at the specified path.

    :param data: Data to save as JSON.
    :param path: Full path to the .json file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# def robust_normalize(df, exclude, path):
#     """
#     Normalize data using robust scaling (median and IQR) from precomputed stats.

#     :param df: DataFrame containing the data to normalize.
#     :param exclude: List of columns to exclude from normalization.
#     :param path: File path to save the computed statistics.
#     :return: Processed DataFrame with normalized data.
#     """
#     fs = s3fs.S3FileSystem(anon=False)
#     newdf = df.copy()

#     stats = get_stats(df)

#     stats_json = json.dumps(stats, indent=4)
#     with fs.open(path, 'w') as f:
#         f.write(stats_json)
    
#     for col in df.columns:
#         if col not in exclude:
#             median = stats[col]['median']
#             iqr = stats[col]['iqr']
            
#             newdf[col] = (df[col] - median) / (iqr if iqr > 0 else 1)

#     return newdf

# def get_stats(df):
#     """
#     Compute mean, standard deviation, median, and IQR for each column in the DataFrame.

#     :param df: DataFrame containing the data to compute statistics for.
#     :return: Dictionary containing statistics for each column.
#     """
#     stats = {}

#     for col in df.columns:
#         series = df[col]

#         mean = series.mean()
#         std = series.std()
#         median = series.median()
#         iqr = series.quantile(0.75) - series.quantile(0.25)

#         stats[col] = {
#             'mean': mean,
#             'std': std,
#             'median': median,
#             'iqr': iqr
#         }

#     return stats