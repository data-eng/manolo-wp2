from enum import Enum


class MlflowSearchParams(Enum):
    EXPERIMENT = "experiment"
    MODEL = "model"
    RUN = "run"
    # METRIC = "metric",
    LATEST_MODEL_VERSION = "latest_model_version"
