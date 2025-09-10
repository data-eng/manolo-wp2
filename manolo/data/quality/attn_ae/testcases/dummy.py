from attn_ae.train import main as train
from attn_ae.infer import main as infer
from attn_ae.utils import *

from converter import create_npz, create_metadata
from torch_loader import
import numpy as np

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

    :return: Tuple of .npz data and .json metadata files.
    """
    T = 10

    features = np.random.rand(T, 2).astype(np.float32)
    time = np.arange(1, T+1).reshape(-1, 1).astype(np.float32)
    labels = np.random.randint(0, 2, size=(T, 1)).astype(np.float32)
    split = labels.copy()
    weights = labels.copy()
    other = np.random.rand(T, 1).astype(np.float32)

    npz_path = get_path('..', 'testcases', 'data', filename="med.npz")
    metadata_path = get_path('..', 'testcases', 'data', filename="med.json")

    create_npz((features, time, labels, split, weights, other), path=npz_path)

    create_metadata(path=metadata_path,
                    keys=(["heart_rate", "blood_pressure", "time", "condition", "night", "gender"],
                          ["HR", "blood_pressure"],
                          ["time"],
                          ["condition"],
                          ["night"],
                          ["condition"],
                          ["gender"])
                    )

    return npz_path, metadata_path

def main():
    npz_path, metadata_path = create_med_dataset()

if __name__ == "__main__":
    main()
