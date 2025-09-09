from typing import TYPE_CHECKING, Optional
from ..enums.DefaultDatastucts import DefaultDatastucts
from ..enums.MlflowSearchParams import MlflowSearchParams
from ..helpers.generic_helpers import GenericHelpers


if TYPE_CHECKING:
    from client import ManoloClient

BASE = "mlflow://"


class MlflowHelper:

    def get_from_mlflow(
            self: "ManoloClient", search_for: MlflowSearchParams = None, id: Optional[str] = None,
            experiment_id: Optional[str] = None, model_name: Optional[str] = None, runId: Optional[str] = None,
            metric_name: Optional[str] = None) -> str:
        """
        Searches MLflow for the requested item: experiment, model, run, metric, or latest model version.

        Args:
            search_for (MlflowSearchParams): The item type to search for.
            id (str, optional): General-purpose ID (usually from ARX).
            experiment_id (str, optional): Specific experiment ID.
            model_name (str, optional): Model name.
            runId (str, optional): Run ID.
            metric_name (str, optional): Metric name/key.

        Returns:
            str: The item data from MLflow.
        """
        if search_for is None:
            return "Please provide a search type (MlflowSearchParams)"

        def resolve_key(*candidates):
            for val in candidates:
                if val:
                    return GenericHelpers.check_id_alias(self, val)
            return None

        base = f"{BASE}{search_for.value}/"

        if search_for == MlflowSearchParams.EXPERIMENT:
            keys = resolve_key(experiment_id, id)
            if not keys:
                return "Please provide an experiment ID"
            return self.get_item_data(DefaultDatastucts.MLFLOW.value, base + keys[0])

        elif search_for == MlflowSearchParams.MODEL:
            keys = resolve_key(model_name, id)
            if not keys:
                return "Please provide a model name"
            return self.get_item_data(DefaultDatastucts.MLFLOW.value, base + keys[0])

        elif search_for == MlflowSearchParams.RUN:
            keys = resolve_key(runId, id)
            if not keys:
                return "Please provide a run ID"
            return self.get_item_data(DefaultDatastucts.MLFLOW.value, base + keys[0])

        elif search_for == MlflowSearchParams.METRIC:
            keys = resolve_key(id+"/"+metric_name)
            if not keys:
                return "Please provide a run ID and metric name"
            return self.get_item_data(DefaultDatastucts.MLFLOW.value, base + f"{keys[0]}/{keys[1]}")

        elif search_for == MlflowSearchParams.LATEST_MODEL_VERSION:
            keys = resolve_key(model_name, id)
            if not keys:
                return "Please provide a model name"
            return self.get_item_data(DefaultDatastucts.MLFLOW.value, base + keys)

        return f"Unsupported search type: {search_for}"

    @staticmethod
    def get_metric_value(response, key: str):
        """Return the value of a metric from an MLflow response."""
        metrics = response.run.data.metrics
        return next((m.value for m in metrics if m.key == key), None)
