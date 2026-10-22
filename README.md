# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

Zidio internship project. RetailPulse analyses a multi-store retail chain to **segment customers, predict churn, forecast demand and optimise inventory**, with MLOps tracking and a Streamlit dashboard.

Data: a synthetic retail chain in Pakistan — 30 stores, 5,000 SKUs, 10,000 customers and ~10M sales lines from 2022 to 2025 (prices in PKR). See [docs/DATASET.md](docs/DATASET.md) for the full data dictionary.

## Progress

| Step | Status | Output |
|---|---|---|
| 0. Setup — repo, environment, folder structure | ✅ Done | `requirements.txt`, project folders |
| 1. Exploratory data analysis | ✅ Done | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (HTML: [`reports/01_eda.html`](reports/01_eda.html)), figures `reports/figures/01_*.png` |
| 2. Feature engineering | ✅ Done | [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb), tables in `data/processed/`, figures `reports/figures/02*.png` |
| 3. Customer segmentation (RFM + K-Means) | ✅ Done | [`notebooks/03_segmentation.ipynb`](notebooks/03_segmentation.ipynb), `data/processed/customer_segments.csv`, `models/kmeans_segments.pkl` |
| 4. Churn prediction (time-split, no leakage) | ⏳ Next | |
| 5. Demand forecasting (Prophet, LSTM, hybrid) | ⏳ | |
| 6. Inventory optimisation | ⏳ | |
| 7. MLOps — MLflow, Optuna, drift detection | ⏳ | |
| 8. Streamlit dashboard | ⏳ | |
| 9–12. API, Power BI, Docker/CI/deployment, final report | ⏳ | |

## Project structure

```
├── notebooks/               # one notebook per step: 01_eda, 02_feature_engineering, ...
├── src/                     # reusable code used by notebooks, tests and the app
│   ├── cleaning.py          #   load + clean sales (DuckDB)
│   ├── features.py          #   customer, daily, weekly, SKU and inventory features
│   └── segmentation.py      #   RFM scores, K-Means pipeline, segment names and actions
├── data/
│   ├── processed/           # feature tables produced by the notebooks
│   └── cache/               # Parquet copies of the sales CSVs (git-ignored, rebuilt automatically)
├── reports/
│   ├── figures/             # all charts, prefixed by step (01_*, 02_*)
│   └── 01_eda.html          # HTML export of the EDA notebook
├── scripts/                 # helper scripts (e.g. regenerate the EDA notebook)
├── retail_clean_dataset/        # main dataset (small CSVs in git, see "Data setup")
├── retail_contaminated_dataset/ # second version of the dataset (used for comparison)
├── docs/DATASET.md          # dataset documentation
├── models/                  # saved models (kmeans_segments.pkl, ...)
├── app/  api/  powerbi/  tests/   # filled in by later steps
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
   - `notebooks/01_eda.ipynb` — builds a Parquet cache in `data/cache/`
   - `notebooks/02_feature_engineering.ipynb` — builds `data/processed/sales_clean.parquet` and the feature tables
   - `notebooks/03_segmentation.ipynb` — customer segments and the K-Means model

The Parquet files are rebuilt automatically and are git-ignored.

## Key findings so far

- **Data quality:** keys, prices and `total_value` are consistent; ~13k exact duplicate lines are removed. Both dataset folders contain injected stockouts and slow movers, each with its own answer key.
- **Customers:** basket value is the same for everyone (~PKR 2,100); customers differ in **how often** they shop (Platinum ≈ 3.3× more baskets than Bronze). No customer ever stops buying, so churn is defined as a **drop in spend** (next quarter vs the same quarter last year).
- **Segments (K-Means, k = 5):** Champions, Loyal, Regulars and Occasional match the Platinum/Gold/Silver/Bronze tiers; the fifth segment, **Slipping Away** (23% of customers), is overdue for a visit. Champions are 10% of customers but 21% of revenue.
- **Demand:** steady growth with strong yearly seasonality (February low, November–December peak). Store-wide promotions cut revenue but barely change units, so forecasts target **units**.
- **Products:** extreme skew — one SKU is ~10% of revenue, and the top 20 SKUs are 22%.
- **Inventory:** the snapshot has a large overstock share; stockout-risk items show zero stock, and slow movers hold 3–6× their reorder point.

Details are in the "Key findings" section of each notebook.
