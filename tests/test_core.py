"""Unit tests for the reusable code in src/."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.cleaning import build_clean_sales
from src.drift import drift_status, psi
from src.inventory import abc_class, eoq, reorder_point, safety_stock, simulate, simulate_naive
from src.metrics import forecast_metrics
from src.segmentation import name_clusters, rfm_segment


def test_cleaning_removes_exact_duplicates_and_bad_quantities(tmp_path):
    rows = pd.DataFrame({"date": ["2025-01-01"] * 4, "receipt_id": ["R1", "R1", "R2", "R3"], "sku_id": ["A", "A", "B", "C"],
                         "quantity": [2, 2, 1, 0], "unit_price": [10.0, 10.0, 5.0, 5.0]})
    src = tmp_path / "sales.csv"
    rows.to_csv(src, index=False)
    log = build_clean_sales(src, tmp_path / "clean.parquet", duckdb.connect())
    assert log == {"raw_rows": 4, "removed_rows": 2, "clean_rows": 2}


def test_psi_zero_for_identical_and_large_for_shifted_data():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 2) > 0.25
    assert [drift_status(v) for v in (0.05, 0.2, 0.5)] == ["Stable", "Moderate", "Significant"]


def test_forecast_metrics_known_values():
    m = forecast_metrics([100, 200], [110, 180])
    assert m["MAE"] == 15
    assert m["WAPE_%"] == pytest.approx(10.0)
    assert m["MAPE_%"] == pytest.approx(10.0)


def test_inventory_formulas():
    ss = safety_stock(1.645, 10, lead_weeks=4)
    assert ss == pytest.approx(32.9)
    assert reorder_point(100, ss, lead_weeks=4) == pytest.approx(432.9)
    # EOQ = sqrt(2 * 52 * 100 * 50 / (0.2 * 10)) = sqrt(260000)
    assert eoq(100, 50, 0.2, 10) == pytest.approx(np.sqrt(260_000))


def test_abc_class_thresholds():
    revenue = pd.Series({"a": 70, "b": 15, "c": 10, "d": 5})
    assert abc_class(revenue).to_dict() == {"a": "A", "b": "A", "c": "B", "d": "C"}


def test_simulation_serves_steady_demand_and_naive_keeps_level():
    demand = [100] * 20
    rec = simulate(demand, rop=150, q=300, start=450)
    assert rec["fill_rate_%"] == 100 and rec["stockout_weeks"] == 0 and rec["orders"] > 0
    naive = simulate_naive(demand, start=200)
    assert naive["fill_rate_%"] == 100 and naive["orders"] == 20


def test_simulation_records_stockouts_when_stock_is_short():
    out = simulate([100, 100, 100], rop=0, q=1, start=250)   # 250 units cover 2.5 weeks
    assert out["stockout_weeks"] == 1 and out["fill_rate_%"] < 100


def test_rfm_rules():
    assert rfm_segment(5, 5, 5) == "Champions"
    assert rfm_segment(1, 1, 1) == "Lost / Dormant"
    assert rfm_segment(1, 4, 4) == "At Risk"


def test_cluster_names_follow_profiles():
    df = pd.DataFrame({"recency_days": [1, 1, 1, 1, 9], "frequency": [900, 700, 400, 300, 350]})
    names = name_clusters(df, labels=[0, 1, 2, 3, 4])
    assert names == {4: "Slipping Away", 0: "Champions", 1: "Loyal", 2: "Regulars", 3: "Occasional"}
