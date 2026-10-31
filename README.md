# RetailPulse

[![CI](https://github.com/Lonishri21/retailpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Lonishri21/retailpulse/actions/workflows/ci.yml)

Customer analytics and demand forecasting for a 30-store retail chain, built as my Zidio internship project.
It segments customers, flags the ones likely to cut their spending, forecasts demand, and works out how much stock to keep.

**Live dashboard:** https://retailpulsegit-fyq3ydw2nxktrp7xckcmjd.streamlit.app/

The data is the [Synthetic Retail Dataset - 10 Million Transactions](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions)
from Kaggle: 30 stores, 5,000 products, 10,000 customers and about 10 million sales lines from 2022 to 2025, with prices in PKR.
More detail in [docs/DATASET.md](docs/DATASET.md).

## What's in the project

| Notebook | What it does |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Explores the data: quality checks, sales trends, stores, products, customers, promotions, stock |
| [02_feature_engineering](notebooks/02_feature_engineering.ipynb) | Cleans the sales and builds customer, daily, product and inventory tables |
| [03_segmentation](notebooks/03_segmentation.ipynb) | Groups customers with RFM scores and K-Means |
| [04_churn](notebooks/04_churn.ipynb) | Predicts which customers will spend less next quarter (XGBoost, explained with SHAP) |
| [05_forecasting](notebooks/05_forecasting.ipynb) | Forecasts units sold with baselines, Prophet, an LSTM and a mix of the two |
| [06_inventory](notebooks/06_inventory.ipynb) | Safety stock, reorder points and order sizes for the top 20 products, tested in a simulation |
| [07_mlops_drift](notebooks/07_mlops_drift.ipynb) | MLflow tracking, Optuna tuning, drift checks and a rule for when to retrain |

The dashboard in `app/` shows all of this across seven pages, and Docker plus GitHub Actions handle deployment and testing.

## Main results

- **Segments:** six customer groups. Four line up with the loyalty tiers; the other two, *Cooling Off* and *Slipping Away*, are customers who are overdue for a visit. Champions are 10% of customers but bring in 21% of revenue.
- **Churn:** nobody actually stops shopping in this data, so churn means next quarter's spend falling more than 30% below the same quarter last year. The model scores a ROC-AUC of 0.85 on a quarter it never saw, and the riskiest 20% of customers include 60% of those who cut back.
- **Forecasting:** the Prophet + LSTM forecast is off by about 2.5% per week on the test period. January 2026 is forecast at 358,000 units, 5.7% up on last January.
- **Inventory:** in a simulation of 2025, the forecast-based stock policy keeps the same service level as a simple "reorder what sold" rule while holding 37% less stock and costing 55% less to run.
- **Drift:** 2025 differs from 2024 mainly because the business grew and ran fewer promotions, so the models should be retrained on a regular schedule.

## Project layout

```
├── notebooks/          one notebook per step (01-07)
├── app/                Streamlit dashboard (Home.py + pages/)
├── src/                code shared by the notebooks, dashboard and tests
│   ├── cleaning.py       load and clean the sales data (DuckDB)
│   ├── features.py       customer, daily, product and inventory features
│   ├── segmentation.py   RFM scores, K-Means, segment names and actions
│   ├── churn.py          churn dataset (features before a date, label after it)
│   ├── forecasting.py    lost-sales estimate, baselines, Prophet, LSTM
│   ├── inventory.py      safety stock, reorder point, EOQ, ABC, simulation
│   ├── metrics.py        forecast error measures
│   ├── drift.py          drift report (PSI and KS test)
│   ├── tracking.py       MLflow setup
│   └── pipeline.py       runs notebooks 02-07 in order
├── data/processed/     tables the notebooks produce and the dashboard reads
├── reports/figures/    charts from the notebooks
├── models/             saved models
├── retail_clean_dataset/  the dataset (small files only, see below)
├── docs/DATASET.md     column reference for the dataset
├── tests/              tests for src/ and every dashboard page
├── Dockerfile
└── requirements.txt    (requirements-lock.txt has exact versions)
```

## Running it

You'll need Python 3.11 (Prophet and TensorFlow work best on it).

```bash
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The sales file is about 770 MB, too big for GitHub, so it isn't in the repo. Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/mrayyanshehzad/synthetic-retail-dataset-10-million-transactions) and copy
`sales_transactions.csv` from its `clean` folder into `retail_clean_dataset/`. Everything else is already here.

Then run the notebooks in order, or let the pipeline do notebooks 02-07 for you (about 25 minutes):

```bash
python -m src.pipeline
```

## Dashboard

```bash
streamlit run app/Home.py
```

It only reads the saved results, so it works without the big sales file. It has its own short `app/requirements.txt`.

Or with Docker:

```bash
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

and open http://localhost:8501.

## Experiment tracking

Notebooks 04, 05 and 07 log every model run to MLflow. The store (`mlflow.db` and `mlruns/`) stays local and isn't
committed, so a copy of the results is kept in `data/processed/mlflow_runs.csv`. To look through the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Testing and deployment

Every push to GitHub runs `ruff` and `pytest`, then builds the Docker image and checks that the container starts.
The dashboard is hosted on Streamlit Community Cloud and redeploys automatically from `main`.

## What I'd add next

- Kubernetes deployment, using the same health check
- An Airflow schedule for the pipeline: weekly for forecasts and stock, monthly for churn
