import os
import pandas as pd
import glob
import mne

import converter

logger = converter.get_logger(level='INFO')

def bitbrain(dir):
    """
    Load the Bitbrain dataset from the specified directory, process it and return a dataframe along with metadata.

    :param dir: Directory containing the Bitbrain dataset.
    :return: A tuple containing the processed dataframe, features, time, labels, and other.
    """
    all_data = []

    features = ['HB_1', 'HB_2']
    time = ['time']
    labels = ['majority']
    other = ['night', 'onset', 'duration', 'begsample', 'endsample', 'offset', 'ai_psg']

    for subject_folder in glob.glob(os.path.join(dir, 'sub-*')):
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
    
    logger.info("Preview of combined dataframe:")
    logger.info(df.head().to_string())

    return df, features, time, labels, other

def main():
    """
    Main function to load the Bitbrain dataset, convert it to numpy format, and save metadata as JSON.
    """
    in_dir = converter.get_dir('..', '..', 'datasets', 'bitbrain_small')
    out_dir = converter.get_dir('..', '..', 'datasets', 'bitbrain_conv')

    out_npy_path = converter.get_path(out_dir, filename='bitbrain.npy')
    out_json_path = converter.get_path(out_dir, filename='bitbrain.json')
    
    logger.info(f"Loading data from {in_dir} and returning it as a dataframe along with metadata.")
    df, features, time, labels, other = bitbrain(dir=in_dir)

    logger.info(f"Converting dataframe to numpy array and saving it as {out_npy_path}.")
    converter.create_npy(df, path=out_npy_path)

    logger.info(f"Creating metadata JSON file at {out_json_path}.")
    converter.create_metadata(
        path=out_json_path,
        column_names=df.columns.tolist(),
        features=features,
        time=time,
        labels=labels,
        other=other
    )

    logger.info(f"Conversion finished! Files (.npy and .json) saved at {out_dir}.")

if __name__ == "__main__":
    main()