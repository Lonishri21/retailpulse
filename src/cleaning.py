"""Load and clean the RetailPulse data.

Shared by the notebooks, tests and the app so that every step cleans data the same way.
All heavy work runs in DuckDB, so the 10M-row sales file is never fully loaded into pandas.
"""
from pathlib import Path

import duckdb

# Small dimension tables: view name -> CSV file name (without .csv)
DIMENSIONS = {
    "stores": "store_master",
    "skus": "sku_master",
    "customers": "customer_master",
    "promos": "promotions",
    "inventory": "inventory_snapshot",
    "flags": "sku_inventory_flags",
}


def build_clean_sales(source: Path, out_path: Path, con: duckdb.DuckDBPyConnection) -> dict:
    """Write the cleaned sales table to parquet and return a cleaning log.

    `source` can be the raw CSV or an existing parquet copy of it.
    Cleaning rules (from the Step 1 EDA):
      - remove exact duplicate rows (the same line recorded twice)
      - drop rows with quantity <= 0 (none expected, kept as a safety net)
    """
    con.execute("SET enable_progress_bar = false")
    source, out_path = Path(source), Path(out_path)
    reader = "read_parquet" if source.suffix == ".parquet" else "read_csv_auto"
    raw = f"{reader}('{source.as_posix()}')"

    n_raw = con.sql(f"SELECT COUNT(*) FROM {raw}").fetchone()[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"""
        COPY (
            SELECT DISTINCT * FROM {raw}
            WHERE quantity > 0
            ORDER BY ALL   -- full sort, so the file (and every later sum) is identical on each run
        ) TO '{out_path.as_posix()}' (FORMAT parquet)
    """)
    n_clean = con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    return {"raw_rows": n_raw, "removed_rows": n_raw - n_clean, "clean_rows": n_clean}


def connect(data_dir: Path, sales_parquet: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with one view per table: sales, stores, skus, customers, promos, inventory, flags."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # One thread adds numbers in a fixed order, so feature values are identical on every run
    # (multi-threaded sums can differ in the last digits). Costs ~10s on the full dataset.
    con.execute("SET threads = 1")
    con.sql(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{Path(sales_parquet).as_posix()}')")
    for view, file in DIMENSIONS.items():
        path = (Path(data_dir) / f"{file}.csv").as_posix()
        con.sql(f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{path}')")
    return con
