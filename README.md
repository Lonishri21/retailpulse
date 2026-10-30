# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

[![CI](https://github.com/Lonishri21/retailpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Lonishri21/retailpulse/actions/workflows/ci.yml)

Zidio internship project: customer segmentation, churn prediction, demand forecasting and inventory optimisation for a multi-store retail chain.

**Live dashboard:** https://retailpulsegit-fyq3ydw2nxktrp7xckcmjd.streamlit.app/

Data: synthetic retail chain, 30 stores, 5,000 SKUs, 10,000 customers, ~10M sales lines (2022–2025, PKR). Source: [Kaggle](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions). See [docs/DATASET.md](docs/DATASET.md).

## Progress

| Step | Status | Output |
|---|---|---|
| 0. Setup — repo, environment, folder structure | ✅ Done | `requirements.txt`, project folders |
| 1. Exploratory data analysis | ✅ Done | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), figures `reports/figures/01_*.png` |
| 2. Feature engineering | ✅ Done | [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb), tables in `data/processed/`, figures `reports/figures/02*.png` |
| 3. Customer segmentation (RFM + K-Means) | ✅ Done | [`notebooks/03_segmentation.ipynb`](notebooks/03_segmentation.ipynb), `data/processed/customer_segments.csv`, `models/kmeans_segments.pkl` |
| 4. Churn prediction (time-split, no leakage) | ✅ Done | [`notebooks/04_churn.ipynb`](notebooks/04_churn.ipynb), `data/processed/customer_churn.csv`, `models/churn_xgb.pkl` |
| 5. Demand forecasting (Prophet, LSTM, hybrid) | ✅ Done | [`notebooks/05_forecasting.ipynb`](notebooks/05_forecasting.ipynb), `data/processed/forecast_*.csv`, `sku_forecasts.csv`, models in `models/` |
| 6. Inventory optimisation | ✅ Done | [`notebooks/06_inventory.ipynb`](notebooks/06_inventory.ipynb), `data/processed/inventory_recommendations.csv`, `inventory_simulation.csv` |
| 7. MLOps — MLflow, Optuna, drift detection | ✅ Done | [`notebooks/07_mlops_drift.ipynb`](notebooks/07_mlops_drift.ipynb), `data/processed/drift_report.csv`, `mlflow_runs.csv`, `src/pipeline.py` |
| 8. Streamlit dashboard | ✅ Done | [`app/`](app/) — 7 pages: overview, sales, customers, churn, forecast, inventory, monitoring |
| 9. FastAPI | ➖ Skipped | |
| 10. Power BI report | ➖ Skipped | |
| 11. Docker, CI, deployment | ✅ Done | `Dockerfile`, `.github/workflows/ci.yml`, `tests/`; [live dashboard](https://retailpulsegit-fyq3ydw2nxktrp7xckcmjd.streamlit.app/) |
| 12. Final report | ⏳ Next | |

## Project structure

```
├── notebooks/               # one notebook per step: 01_eda, 02_feature_engineering, ...
├── app/                     # Streamlit dashboard (Home.py + pages/), reads saved outputs only
├── src/                     # reusable code used by notebooks, tests and the app
│   ├── cleaning.py          #   load + clean sales (DuckDB)
│   ├── features.py          #   customer, daily, weekly, SKU and inventory features
│   ├── segmentation.py      #   RFM scores, K-Means pipeline, segment names and actions
│   ├── churn.py             #   churn dataset: features before a cutoff, label after it
│   ├── forecasting.py       #   lost-sales estimate, baselines, Prophet, LSTM
│   ├── inventory.py         #   safety stock, reorder point, EOQ, ABC, stock simulation
│   ├── drift.py             #   PSI + KS drift report
│   ├── tracking.py          #   MLflow setup (local SQLite store)
│   ├── pipeline.py          #   runs all steps in order: python -m src.pipeline
│   └── metrics.py           #   forecast accuracy metrics (WAPE, MAPE, MAE, RMSE)
├── data/
│   ├── processed/           # feature tables produced by the notebooks
│   └── cache/               # Parquet copies of the sales CSVs (git-ignored, rebuilt automatically)
├── reports/
│   └── figures/             # all charts, prefixed by step (01_*, 02_*, ...)
├── retail_clean_dataset/        # main dataset (small CSVs in git, see "Data setup")
├── retail_contaminated_dataset/ # second version of the dataset (used for comparison)
├── docs/DATASET.md          # dataset documentation
├── models/                  # saved models (segments, churn, Prophet, LSTM)
├── tests/                   # unit tests (src) + smoke tests (every dashboard page)
├── .github/workflows/ci.yml # lint + tests + Docker build on every push
├── Dockerfile               # dashboard container
├── requirements.txt         # dependencies
└── requirements-lock.txt    # exact versions
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Python 3.11 is recommended (Prophet and TensorFlow are most reliable on it).

## Data setup

The two `sales_transactions.csv` files (~770 MB each) are **not stored in git** because GitHub limits files to 100 MB. All other CSVs are included.

1. Download [Synthetic Retail Dataset - 10 Million Transactions](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions) from Kaggle
2. Put the large files at:
   - `retail_clean_dataset/sales_transactions.csv`
   - `retail_contaminated_dataset/sales_transactions.csv`
3. Run the notebooks in order:
   - `notebooks/01_eda.ipynb` — builds a Parquet cache in `data/cache/`
   - `notebooks/02_feature_engineering.ipynb` — builds `data/processed/sales_clean.parquet` and the feature tables
   - `notebooks/03_segmentation.ipynb` — customer segments and the K-Means model
   - `notebooks/04_churn.ipynb` — churn model and today's risk scores (~3 min)
   - `notebooks/05_forecasting.ipynb` — forecast comparison, 30-day and SKU forecasts (~5 min)
   - `notebooks/06_inventory.ipynb` — inventory recommendations and simulation
   - `notebooks/07_mlops_drift.ipynb` — Optuna tuning, drift report, monitoring (~10 min)

   Or run steps 2–7 in one go: `python -m src.pipeline`.

The Parquet files are rebuilt automatically and are git-ignored.

## Dashboard

```bash
streamlit run app/Home.py
```
The app only reads files in `data/processed/`, `reports/figures/` and the small dataset CSVs, so it runs without the large sales files. It has its own light `app/requirements.txt`.

## Deployment

**Live dashboard:** https://retailpulsegit-fyq3ydw2nxktrp7xckcmjd.streamlit.app/

**Docker** (only the dashboard and the small files it reads go into the image):
```bash
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```
then open http://localhost:8501.

**Continuous integration:** every push runs `ruff` and `pytest` (unit tests for `src/` and a smoke test of every dashboard page), then builds the Docker image and checks the container's health endpoint.

**Streamlit Community Cloud:** at share.streamlit.io, sign in with GitHub → *Create app* → repo `Lonishri21/retailpulse`, branch `main`, main file `app/Home.py` → *Advanced settings*: Python 3.11 → *Deploy*. Cloud installs the light `app/requirements.txt` next to the main file.

**Future work:** Kubernetes (2 replicas behind a Service, using the same health check as a liveness probe) and an Airflow DAG for the pipeline.

## Experiment tracking

Notebooks 04, 05 and 07 log every model run to a local MLflow store (`mlflow.db` + `mlruns/`, git-ignored). To browse runs and the registered `churn_xgb` model:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
then open http://127.0.0.1:5000. A copy of the run table is kept in `data/processed/mlflow_runs.csv`.
