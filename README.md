# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

Zidio internship project. RetailPulse analyses a multi-store retail chain to **segment customers, predict churn, forecast demand and optimise inventory**, with MLOps tracking and a Streamlit dashboard.

Data: a synthetic retail chain in Pakistan — 30 stores, 5,000 SKUs, 10,000 customers and ~10M sales lines from 2022 to 2025 (prices in PKR). See [docs/DATASET.md](docs/DATASET.md) for the full data dictionary.

## Progress

| Step | Status | Output |
|---|---|---|
| 0. Setup — repo, environment, folder structure | ✅ Done | `requirements.txt`, project folders |
| 1. Exploratory data analysis | ✅ Done | [`01_data_analysis/eda.ipynb`](01_data_analysis/eda.ipynb) (HTML: `eda.html`), figures in `01_data_analysis/figures/` |
| 2. Feature engineering | ✅ Done | [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb), tables in `data/processed/`, figures in `reports/figures/` |
| 3. Customer segmentation (RFM + K-Means) | ⏳ Next | |
| 4. Churn prediction (time-split, no leakage) | ⏳ | |
| 5. Demand forecasting (Prophet, LSTM, hybrid) | ⏳ | |
| 6. Inventory optimisation | ⏳ | |
| 7. MLOps — MLflow, Optuna, drift detection | ⏳ | |
| 8. Streamlit dashboard | ⏳ | |
| 9–12. API, Power BI, Docker/CI/deployment, final report | ⏳ | |

## Project structure

```
├── 01_data_analysis/        # Step 1: EDA notebook, HTML export, figures
├── notebooks/               # Step 2 onwards, one notebook per step
├── src/                     # reusable code used by notebooks, tests and the app
│   ├── cleaning.py          #   load + clean sales (DuckDB)
│   └── features.py          #   customer, daily, weekly, SKU and inventory features
├── data/processed/          # feature tables produced by the notebooks
├── reports/figures/         # charts from Step 2 onwards
├── retail_clean_dataset/        # main dataset (small CSVs in git, see "Data setup")
├── retail_contaminated_dataset/ # second version of the dataset (used for comparison)
├── docs/DATASET.md          # dataset documentation
├── models/  app/  api/  powerbi/  tests/   # filled in by later steps
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

1. Download the dataset from: **<add dataset download link here>**
2. Put the large files at:
   - `retail_clean_dataset/sales_transactions.csv`
   - `retail_contaminated_dataset/sales_transactions.csv`
3. Run the notebooks in order:
   - `01_data_analysis/eda.ipynb` — builds a Parquet cache in `01_data_analysis/cache/`
   - `notebooks/02_feature_engineering.ipynb` — builds `data/processed/sales_clean.parquet` and the feature tables

The Parquet files are rebuilt automatically and are git-ignored.

## Key findings so far

- **Data quality:** keys, prices and `total_value` are consistent; ~13k exact duplicate lines are removed. Both dataset folders contain injected stockouts and slow movers, each with its own answer key.
- **Customers:** basket value is the same for everyone (~PKR 2,100); customers differ in **how often** they shop (Platinum ≈ 3.3× more baskets than Bronze). No customer ever stops buying, so churn is defined as a **drop in spend** (next quarter vs the same quarter last year).
- **Demand:** steady growth with strong yearly seasonality (February low, November–December peak). Store-wide promotions cut revenue but barely change units, so forecasts target **units**.
- **Products:** extreme skew — one SKU is ~10% of revenue, and the top 20 SKUs are 22%.
- **Inventory:** the snapshot has a large overstock share; stockout-risk items show zero stock, and slow movers hold 3–6× their reorder point.

Details are in the "Key findings" section of each notebook.
