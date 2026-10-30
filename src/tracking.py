"""MLflow setup: a local SQLite store (needed for the model registry) at the project root."""
import os
from pathlib import Path

import logging
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
import mlflow  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)
# classifier outputs are integer labels; this schema hint does not apply to our inputs
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def use_experiment(name: str):
    """Point MLflow at the project store and select (or create) an experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return mlflow
