"""MLflow helpers. Runs go to mlflow.db at the project root."""
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
import mlflow  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)
# mlflow warns about int columns in the model output; not an issue here
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def use_experiment(name: str):
    """Use the project store and switch to (or create) an experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return mlflow
