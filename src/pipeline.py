"""Run the RetailPulse steps in order: clean + features -> segment -> churn -> forecast -> inventory -> MLOps/drift.

Each step is its notebook, executed top to bottom (outputs are saved back into the notebook).
If a step fails, the pipeline stops so later steps never run on stale inputs.

    python -m src.pipeline              # run every step
    python -m src.pipeline --from churn # start at a later step
    python -m src.pipeline --list       # show the steps
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("features", "02_feature_engineering.ipynb", "clean sales + feature tables"),
    ("segment", "03_segmentation.ipynb", "RFM + K-Means segments"),
    ("churn", "04_churn.ipynb", "churn model + risk scores"),
    ("forecast", "05_forecasting.ipynb", "Prophet / LSTM / hybrid forecasts"),
    ("inventory", "06_inventory.ipynb", "safety stock, reorder points, EOQ"),
    ("mlops", "07_mlops_drift.ipynb", "tuning, drift report, monitoring"),
]


def run_notebook(path: Path, timeout: int = 3600) -> None:
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], default=STEPS[0][0])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for i, (name, nb, what) in enumerate(STEPS, 1):
            print(f"{i}. {name:<10} {nb:<30} {what}")
        return 0
    if not (ROOT / "retail_clean_dataset" / "sales_transactions.csv").exists():
        print("Missing retail_clean_dataset/sales_transactions.csv - see 'Data setup' in README.md")
        return 1
    names = [s[0] for s in STEPS]
    for name, nb, what in STEPS[names.index(args.start):]:
        t = time.time()
        print(f"-> {name}: {what} ...", flush=True)
        try:
            run_notebook(ROOT / "notebooks" / nb)
        except Exception as e:  # stop the pipeline on the first failure
            print(f"FAILED at step '{name}' ({nb}): {type(e).__name__}: {str(e)[:500]}")
            return 1
        print(f"   done in {time.time() - t:.0f}s")
    print("pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
