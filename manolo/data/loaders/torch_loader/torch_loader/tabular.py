import numpy as np

from . import utils

logger = utils.get_logger(level='DEBUG')

def get_stats(dir, name):
    """
    Load structured .npz train data and metadata, compute stats (mean, std, median, IQR) per column,
    and save the stats as a JSON file.

    :param dir: Directory containing {name}-train.npz and {name}.json.
    :param name: Dataset name prefix (e.g., 'bitbrain').
    :return: Dict of stats keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-train.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npz(data_path)
    metadata = utils.load_json(meta_path)

    columns_values = data["columns"]
    column_names = metadata["columns"]

    stats = {}

    for i, col in enumerate(column_names):
        col_values = columns_values[:, i]

        mean = np.mean(col_values)
        std = np.std(col_values)
        median = np.median(col_values)
        q75, q25 = np.percentile(col_values, [75, 25])
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

def robust_normalize(dir, name, process, include, stats):
    """
    Normalize structured .npz dataset using robust scaling (median and IQR) from precomputed stats. Applies normalization only to specified columns across any sub-array.

    :param dir: Directory containing the dataset.
    :param name: Dataset base name (e.g., 'bitbrain').
    :param process: Process type (e.g., 'train', 'val', 'test').
    :param include: List of column names to include in normalization.
    :param stats: Dict of precomputed stats (median, iqr) keyed by column name.
    """
    data_path = utils.get_path(dir, filename=f"{name}-{process}.npz")
    meta_path = utils.get_path(dir, filename=f"{name}.json")

    data = utils.load_npy(data_path)
    metadata = utils.load_json(meta_path)

    all_columns = metadata["columns"]
    data_norm = {k: v.copy().astype(np.float32) for k, v in data.items()}

    for col in include:
        median = stats[col]["median"]
        iqr = stats[col]['iqr'] if stats[col]['iqr'] > 0 else 1.0

        data_
        

    #         median = stats[col]['median']
    #         iqr = stats[col]['iqr'] if stats[col]['iqr'] > 0 else 1.0

    #         data_norm[:, idx] = (data[:, idx] - median) / iqr

    # norm_path = utils.get_path(dir, filename=f"{name}-{process}-robust-norm.npy")
    # utils.save_npy(data_norm, norm_path)

    # logger.info(f"Robust normalized data saved to {norm_path}.")

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