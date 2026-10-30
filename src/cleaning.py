"""Load and clean the sales data with DuckDB."""
from pathlib import Path

import duckdb

# view name -> CSV file name
DIMENSIONS = {
    "stores": "store_master",
    "skus": "sku_master",
    "customers": "customer_master",
    "promos": "promotions",
    "inventory": "inventory_snapshot",
    "flags": "sku_inventory_flags",
}


def build_clean_sales(source: Path, out_path: Path, con: duckdb.DuckDBPyConnection) -> dict:
    """Drop exact duplicates and rows with quantity <= 0, write parquet, return row counts."""
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
            ORDER BY ALL
        ) TO '{out_path.as_posix()}' (FORMAT parquet)
    """)
    n_clean = con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    return {"raw_rows": n_raw, "removed_rows": n_raw - n_clean, "clean_rows": n_clean}


def connect(data_dir: Path, sales_parquet: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with a view for each table."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # one thread so sums come out the same on every run
    con.execute("SET threads = 1")
    con.sql(f"CREATE VIEW sales AS SELECT * FROM read_parquet('{Path(sales_parquet).as_posix()}')")
    for view, file in DIMENSIONS.items():
        path = (Path(data_dir) / f"{file}.csv").as_posix()
        con.sql(f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{path}')")
    return con
