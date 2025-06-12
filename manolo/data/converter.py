import os
import json
import numpy as np
import pandas as pd
import logging
import glob
import mne

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
    all_data = []

    for subject_folder in glob.glob(os.path.join(in_dir, 'sub-*')):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = os.path.join(subject_folder, 'eeg')

        if not os.path.exists(eeg_folder):
            continue

        eeg_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-headband_eeg.edf')
        events_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-psg_events.tsv')

        try:
            raw = mne.io.read_raw_edf(eeg_file_pattern, preload=True)
            x_data = raw.to_data_frame()
        except Exception as e:
            continue

        try:
            y_data = pd.read_csv(events_file_pattern, delimiter='\t')
        except Exception as e:
            continue

        y_expanded = pd.DataFrame(index=x_data.index, columns=y_data.columns)

        for _, row in y_data.iterrows():
            begsample = row['begsample'] - 1
            endsample = row['endsample'] - 1

            y_expanded.loc[begsample:endsample] = row.values

        combined_data = pd.concat([x_data, y_expanded], axis=1)
        combined_data['night'] = int(subject_id.replace('sub-', ''))

        all_data.append(combined_data)
        df = pd.concat(all_data, ignore_index=True)
    
    print("Preview of combined dataframe:")
    print(df.head().to_string())

    time_cols = ['time']
    label_cols = ['majority']
    feature_cols = ['HB_1', 'HB_2']
    other_cols = ['night', 'onset', 'duration', 'begsample', 'endsample', 'offset', 'ai_psg']

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

    in_dir = get_dir('..', '..', 'datasets', 'bitbrain_small')
    out_dir = get_dir('..', '..', 'datasets', 'bitbrain_conv')
    
    logger.info("Converting dataset to npy and creating its metadata.")
    bitbrain(in_dir, out_dir)

    logger.info(f"Conversion finished! Files saved at {out_dir}.")

if __name__ == "__main__":
    main()