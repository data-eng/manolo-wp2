import os
import json
import numpy as np
import pandas as pd
import logging

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
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

def create_metadata(path, column_names, features, time, labels, other):
    metadata = {
        "columns": column_names,
        "features": features,
        "time": time,
        "labels": labels,
        "other": other
    }

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def convert_to_npy(df, path):
    np.save(path, df.values.astype(np.float32))

def bitbrain(in_dir, out_dir):
    path = get_path(in_dir, filename='test_unorm.csv')
    df = pd.read_csv(path)

    time_cols = ['time']
    label_cols = ['majority']
    feature_cols = ['HB_1', 'HB_2']
    other_cols = ['seq_id', 'night']

    out_npy_path = get_path(out_dir, filename='bitbrain.npy')
    convert_to_npy(df, path=out_npy_path)

    out_json_path = get_path(out_dir, filename='bitbrain.json')
    create_metadata(
        path=out_json_path,
        column_names=df.columns.tolist(),
        features=feature_cols,
        time=time_cols,
        labels=label_cols,
        other=other_cols
    )

def main():
    logger = get_logger(level='INFO')

    in_dir = get_dir('..', '..', 'data', 'proc')
    out_dir = get_dir('..', '..', 'data', 'bitbrain_conv')
    
    logger.info("Converting dataset to npy and creating its metadata.")
    bitbrain(in_dir, out_dir)

    logger.info(f"Conversion finished! Files saved at {out_dir}.")

if __name__ == "__main__":
    main()