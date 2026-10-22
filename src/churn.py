"""Churn dataset: features before a cutoff, label from the quarter after it."""
import pandas as pd

from src.features import customer_features

DROP_RATIO = 0.7          # churn = next-quarter list-price spend < 70% of the same quarter last year
TIERS = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

CHURN_FEATURES = [
    "recency_days", "spend_30", "spend_90", "spend_prev_90", "spend_365", "visits_30", "visits_90", "visits_prev_90",
    "avg_days_between", "max_gap_days", "gap_std_days", "overdue_ratio", "aov", "items_per_basket",
    "spend_trend_90", "visits_trend_90", "promo_line_share", "discount_share", "preferred_channel_share", "age",
    "list_spend_same_q_ly", "list_spend_last_q", "last_q_yoy", "loyalty_tier", "target_quarter",
]


def _list_spend(con, start, end) -> pd.Series:
    """Spend at list price (before discounts) per customer in [start, end)."""
    return con.sql(f"""
        SELECT customer_id, SUM(quantity * unit_price) AS s FROM sales
        WHERE date >= DATE '{start:%Y-%m-%d}' AND date < DATE '{end:%Y-%m-%d}' GROUP BY 1
    """).df().set_index("customer_id")["s"]


def churn_table(con, cutoff, with_label: bool = True) -> pd.DataFrame:
    """One row per customer: features from sales before `cutoff`, label from the 3 months after it."""
    c, q = pd.Timestamp(cutoff), pd.DateOffset(months=3)
    f = customer_features(con, c)
    f["list_spend_same_q_ly"] = _list_spend(con, c - 4 * q, c - 3 * q).reindex(f.index).fillna(0)
    f["list_spend_last_q"] = _list_spend(con, c - q, c).reindex(f.index).fillna(0)
    prev = _list_spend(con, c - 5 * q, c - 4 * q).reindex(f.index)
    f["last_q_yoy"] = f["list_spend_last_q"] / prev - 1
    f["loyalty_tier"] = f["loyalty_segment"].map(TIERS)
    f["target_quarter"] = c.quarter
    f["cutoff"] = c
    if with_label:
        f["list_spend_next_q"] = _list_spend(con, c, c + q).reindex(f.index).fillna(0)
        f["churn"] = (f["list_spend_next_q"] < DROP_RATIO * f["list_spend_same_q_ly"]).astype(int)
    return f
