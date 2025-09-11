from attn_ae.train import main as train
from attn_ae.infer import main as infer
from attn_ae.utils import *

from converter import create_npz, create_metadata
from torch_loader import preprocess

import numpy as np
import requests

logger = get_logger(level='INFO')

def create_med_dataset():
    """
    Generate a small, synthetic medical dataset for testing purposes.

    The dataset includes:
      - 2 input features representing medical signals (e.g., heart rate, blood pressure).
      - 1 target label for supervised tasks.
      - A sequential time column.
      - A split column (used to indicate subsets for training/testing).
      - A weights column (for sample weighting).
      - An additional 'other' feature not used in model input.
    """
    T = 1000

    features = np.random.rand(T, 2).astype(np.float32)
    time = np.arange(1, T+1).reshape(-1, 1).astype(np.float32)
    labels = np.random.randint(0, 2, size=(T, 1)).astype(np.float32)
    
    split = np.random.randint(0, 5, size=(T, 1)).astype(np.float32)
    weights = labels.copy()
    other = np.random.rand(T, 1).astype(np.float32)

    npz_path = get_path('..', 'testcases', 'data', filename='med.npz')
    metadata_path = get_path('..', 'testcases', 'data', filename='med.json')

    create_npz((features, time, labels, split, weights, other), path=npz_path)

    create_metadata(path=metadata_path,
                    keys=(['heart_rate', 'blood_pressure', 'time', 'condition', 'night', 'gender'],
                          ['HR', 'blood_pressure'],
                          ['time'],
                          ['condition'],
                          ['night'],
                          ['condition'],
                          ['gender'])
                    )

def prepare_params(ds_dir):
    for process in ('prepare', 'work'):
        loaders = preprocess(
            dir=ds_dir,
            name='med',
            process=process,
            train_size=0.65,
            val_size=0.15,
            infer_size=0.20,
            seq_len=3,
            norm_include=['heart_rate', 'blood_pressure', 'time'],
            full_epoch=6,
            per_epoch=False,
            time_include=True,
            shifted=False,
            splitted=False,
            weighted=False,
            analyzed=False,
            normalized=False,
            weights_from='train',
            stats_from='train'
        )

        weights_path = get_path(ds_dir, filename=f"med-weights.json")
        weights = load_json(weights_path)

        if process == 'prepare':
            save_url = get_path('..', 'testcases', 'models', filename=f'attn_ae.pth')

            train_data_id_params = {"data": loaders, "weights": weights}

            train_model_params = {
                "model_params": {
                    "seq_len": 3,
                    "num_feats": 3,
                    "latent_seq_len": 1,
                    "latent_num_feats": 6,
                    "num_heads": 1,
                    "num_layers": 1,
                    "dropout": 0.05
                },
                "model_pth": save_url
            }

            train_options_params = {
                "process_params": {
                    "batch_size": 8,
                    "loss": "BlendedLoss",
                    "epochs": 3,
                    "patience": 30,
                    "lr": 0.0001,
                    "optimizer": "Adam",
                    "scheduler": {
                        "name": "ReduceLROnPlateau",
                        "params": {
                            "factor": 0.99,
                            "patience": 3
                        }
                    }
                },
                "metrics": [
                    "epochs",
                    "train_time",
                    "best_train_loss",
                    "best_val_loss"
                ]
            }
        
        else:
            model_url = get_path('..', 'testcases', 'models', filename=f'attn_ae.zip')

            infer_data_id_params = {"data": loaders, "weights": weights}
            infer_model_params = model_url

            infer_options_params = {
                "process_params": {
                    "batch_size": 8,
                    "loss": "BlendedLoss"
                },
                "metrics": [
                    "infer_loss",
                    "mae",
                    "mse"
                ]
            }

        train_data_id_params_path = get_path('..', 'testcases', 'models', filename=f"train_data_id.pkl")
        train_model_params_path = get_path('..', 'testcases', 'models', filename=f"train_model.pkl")
        train_options_params_path = get_path('..', 'testcases', 'models', filename=f"train_options.pkl")

        save_pickle(train_data_id_params, train_data_id_params_path)
        save_pickle(train_model_params, train_model_params_path)
        save_pickle(train_options_params, train_options_params_path)

        infer_data_id_params_path = get_path('..', 'testcases', 'models', filename=f"infer_data_id.pkl")
        infer_model_params_path = get_path('..', 'testcases', 'models', filename=f"infer_model.pkl")
        infer_options_params_path = get_path('..', 'testcases', 'models', filename=f"infer_options.pkl")

        save_pickle(infer_data_id_params, infer_data_id_params_path)
        save_pickle(infer_model_params, infer_model_params_path)
        save_pickle(infer_options_params, infer_options_params_path)

        train_param_files = {
            "data_id": train_data_id_params_path,
            "model": train_model_params_path,
            "options": train_options_params_path
        }
    
        infer_param_files = {
            "data_id": infer_data_id_params_path,
            "model": infer_model_params_path,
            "options": infer_options_params_path
        }

        return train_param_files, infer_param_files

def remote_call(package, method, param_files):
    response = requests.get(
        'http://127.0.0.1:48033/run',
        params={
            "package": package, 
            "method": method, 
            **param_files
            }
    )

    api_response = response.json()
    
    return api_response["results"]

def main():
    ds_dir = get_dir('..', 'testcases', 'data')
    create_med_dataset()

    train_param_files, infer_param_files = prepare_params(ds_dir)

    remote_call(package='attn_ae', method='train', **train_param_files)
    remote_call(package='attn_ae', method='infer', **infer_param_files)

    logger.info("Training and inference complete!")

if __name__ == "__main__":
    main()
