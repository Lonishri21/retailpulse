"""Feature tables. Functions with a `snapshot` use only sales before that date."""
import numpy as np
import pandas as pd


def _date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def customer_features(con, snapshot) -> pd.DataFrame:
    """One row per customer, using only sales before `snapshot`."""
    S = f"DATE '{_date(snapshot)}'"

    base = con.sql(f"""
        WITH tx AS (
            SELECT s.*, k.category FROM sales s JOIN skus k USING (sku_id)
            WHERE s.date < {S}
        ),
        visits AS (
            SELECT customer_id, receipt_id, MIN(date) AS date,
                   ANY_VALUE(channel) AS channel, SUM(total_value) AS spend
            FROM tx GROUP BY 1, 2
        ),
        v AS (
            SELECT customer_id,
                date_diff('day', MAX(date), {S})                   AS recency_days,
                date_diff('day', MIN(date), {S})                   AS tenure_days,
                COUNT(*)                                           AS frequency,
                SUM(spend)                                         AS monetary,
                SUM(spend) FILTER (WHERE date >= {S} - 30)         AS spend_30,
                SUM(spend) FILTER (WHERE date >= {S} - 90)         AS spend_90,
                SUM(spend) FILTER (WHERE date >= {S} - 180
                                     AND date <  {S} - 90)         AS spend_prev_90,
                SUM(spend) FILTER (WHERE date >= {S} - 180)        AS spend_180,
                SUM(spend) FILTER (WHERE date >= {S} - 365)        AS spend_365,
                COUNT(*)   FILTER (WHERE date >= {S} - 30)         AS visits_30,
                COUNT(*)   FILTER (WHERE date >= {S} - 90)         AS visits_90,
                COUNT(*)   FILTER (WHERE date >= {S} - 180
                                     AND date <  {S} - 90)         AS visits_prev_90,
                AVG((channel = 'In-Store')::INT)                   AS share_in_store,
                AVG((channel = 'Online')::INT)                     AS share_online,
                AVG((channel = 'Mobile App')::INT)                 AS share_mobile_app
            FROM visits GROUP BY 1
        ),
        gaps AS (
            SELECT customer_id, MAX(gap) AS max_gap_days, STDDEV(gap) AS gap_std_days
            FROM (
                SELECT customer_id,
                       date_diff('day', LAG(d) OVER (PARTITION BY customer_id ORDER BY d), d) AS gap
                FROM (SELECT DISTINCT customer_id, date AS d FROM visits)
            ) GROUP BY 1
        ),
        lines AS (
            SELECT customer_id,
                COUNT(*)                              AS n_lines,
                SUM(quantity)                         AS units,
                COUNT(DISTINCT sku_id)                AS unique_skus,
                AVG((promo_id IS NOT NULL)::INT)      AS promo_line_share,
                1 - SUM(total_value) / SUM(quantity * unit_price) AS discount_share
            FROM tx GROUP BY 1
        )
        SELECT * FROM v JOIN gaps USING (customer_id) JOIN lines USING (customer_id)
    """).df().set_index("customer_id")

    # share of spend per category
    cat = con.sql(f"""
        SELECT customer_id, k.category, SUM(total_value) AS spend
        FROM sales s JOIN skus k USING (sku_id)
        WHERE s.date < {S} GROUP BY 1, 2
    """).df().pivot(index="customer_id", columns="category", values="spend").fillna(0)
    cat = cat.div(cat.sum(axis=1), axis=0)
    cat.columns = ["cat_" + c.lower().replace(" & ", "_").replace(" ", "_") for c in cat.columns]

    demo = con.sql("""
        SELECT cust_id AS customer_id, age, gender, city, loyalty_segment, preferred_channel, registration_date
        FROM customers
    """).df().set_index("customer_id")

    f = base.join(cat).join(demo)
    counts = [c for c in f.columns if c.startswith(("spend_", "visits_"))]
    f[counts] = f[counts].fillna(0)

    f["aov"] = f["monetary"] / f["frequency"]
    f["items_per_basket"] = f["n_lines"] / f["frequency"]
    f["units_per_basket"] = f["units"] / f["frequency"]
    f["avg_days_between"] = (f["tenure_days"] - f["recency_days"]) / (f["frequency"] - 1).replace(0, np.nan)
    f["overdue_ratio"] = f["recency_days"] / f["avg_days_between"]
    f["spend_trend_90"] = f["spend_90"] / f["spend_prev_90"].replace(0, np.nan) - 1
    f["visits_trend_90"] = f["visits_90"] / f["visits_prev_90"].replace(0, np.nan) - 1
    share_col = {"In-Store": "share_in_store", "Online": "share_online", "Mobile App": "share_mobile_app"}
    f["preferred_channel_share"] = [f.at[i, share_col[ch]] for i, ch in zip(f.index, f["preferred_channel"])]
    f["days_since_registration"] = (pd.Timestamp(snapshot) - pd.to_datetime(f["registration_date"])).dt.days
    return f.drop(columns=["registration_date"]).sort_index()


def daily_sales(con) -> pd.DataFrame:
    """Chain-level daily totals, promo calendar and stockout counts."""
    d = con.sql("""
        WITH days AS (
            SELECT date,
                SUM(total_value)                 AS revenue,
                SUM(quantity)                    AS units,
                SUM(total_value - quantity * k.cost_price) AS gross_profit,
                COUNT(DISTINCT receipt_id)       AS receipts,
                COUNT(*)                         AS lines,
                COUNT(DISTINCT customer_id)      AS customers
            FROM sales s JOIN skus k USING (sku_id) GROUP BY 1
        ),
        promo_cal AS (
            SELECT d.date,
                COUNT(p.promo_id)                                            AS active_promos,
                COALESCE(MAX((p.target_type = 'All')::INT), 0)               AS store_wide_promo,
                COALESCE(MAX(p.discount_pct) FILTER (WHERE p.target_type = 'All'), 0) AS store_wide_discount
            FROM days d LEFT JOIN promos p ON d.date BETWEEN p.start_date AND p.end_date
            GROUP BY 1
        ),
        pairs AS (
            SELECT sku_id, UNNEST(string_split(affected_stores, ';')) AS store_id,
                   CAST(window_start AS DATE) AS ws, CAST(window_end AS DATE) AS we
            FROM flags WHERE flag = 'STOCKOUT_RISK'
        ),
        stockouts AS (
            SELECT d.date, COUNT(p.sku_id) AS stockout_pairs
            FROM days d LEFT JOIN pairs p ON d.date BETWEEN p.ws AND p.we GROUP BY 1
        )
        SELECT * FROM days JOIN promo_cal USING (date) JOIN stockouts USING (date) ORDER BY date
    """).df()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").asfreq("D", fill_value=0)
    d["aov"] = (d["revenue"] / d["receipts"]).fillna(0)
    return d


def add_time_features(daily: pd.DataFrame, target: str = "units") -> pd.DataFrame:
    """Calendar, lag and rolling features (shifted, no same-day values)."""
    f = daily.copy()
    idx = f.index
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["quarter"] = idx.quarter
    f["year"] = idx.year
    f["dayofyear"] = idx.dayofyear
    f["weekofyear"] = idx.isocalendar().week.astype(int).values
    f["days_to_month_end"] = idx.days_in_month - idx.day
    f["is_nov_dec"] = idx.month.isin([11, 12]).astype(int)
    f["is_feb"] = (idx.month == 2).astype(int)
    f["trend"] = (idx - idx[0]).days
    for lag in [1, 7, 14, 28, 364]:
        f[f"{target}_lag_{lag}"] = f[target].shift(lag)
    past = f[target].shift(1)
    for w in [7, 28]:
        f[f"{target}_roll_mean_{w}"] = past.rolling(w).mean()
        f[f"{target}_roll_std_{w}"] = past.rolling(w).std()
    return f


def weekly_sales(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly Mon-Sun totals, full weeks only."""
    cols = ["revenue", "units", "gross_profit", "receipts", "lines", "store_wide_promo", "stockout_pairs"]
    w = daily[cols].resample("W-SUN").sum()
    w["days"] = daily["units"].resample("W-SUN").count()
    w = w[w["days"] == 7].drop(columns="days")
    return w.rename(columns={"store_wide_promo": "store_wide_promo_days",
                             "stockout_pairs": "stockout_store_sku_days"})


def sku_weekly(con, top_n: int = 20) -> pd.DataFrame:
    """Weekly demand for the top-N SKUs by revenue."""
    w = con.sql(f"""
        WITH top AS (
            SELECT sku_id, ROW_NUMBER() OVER (ORDER BY SUM(total_value) DESC) AS revenue_rank
            FROM sales GROUP BY 1 ORDER BY revenue_rank LIMIT {top_n}
        )
        SELECT sku_id, CAST(date_trunc('week', date) AS DATE) + 6 AS week_end,
               SUM(quantity) AS units, SUM(total_value) AS revenue,
               COUNT(DISTINCT receipt_id) AS receipts,
               AVG((promo_id IS NOT NULL)::INT) AS promo_line_share,
               SUM(quantity * discount_pct) / SUM(quantity) AS avg_discount_pct
        FROM sales JOIN top USING (sku_id) GROUP BY 1, 2
    """).df()
    w["week_end"] = pd.to_datetime(w["week_end"])

    so = con.sql("""
        WITH pairs AS (
            SELECT sku_id, UNNEST(string_split(affected_stores, ';')) AS store_id,
                   CAST(window_start AS DATE) AS ws, CAST(window_end AS DATE) AS we
            FROM flags WHERE flag = 'STOCKOUT_RISK'
        ), days AS (
            SELECT sku_id, store_id, UNNEST(generate_series(ws, we, INTERVAL 1 DAY)) AS d FROM pairs
        )
        SELECT sku_id, CAST(date_trunc('week', d) AS DATE) + 6 AS week_end, COUNT(*) AS stockout_store_days
        FROM days GROUP BY 1, 2
    """).df()
    so["week_end"] = pd.to_datetime(so["week_end"])

    # zero-filled SKU x week grid
    first, last = con.sql("SELECT MIN(date), MAX(date) FROM sales").fetchone()
    weeks = pd.date_range(pd.Timestamp(first) + pd.Timedelta(days=6), pd.Timestamp(last), freq="W-SUN")
    weeks = weeks[weeks - pd.Timedelta(days=6) >= pd.Timestamp(first)]
    grid = pd.MultiIndex.from_product([w["sku_id"].unique(), weeks], names=["sku_id", "week_end"])
    out = (w.set_index(["sku_id", "week_end"]).reindex(grid)
             .join(so.set_index(["sku_id", "week_end"])).reset_index())
    fill0 = ["units", "revenue", "receipts", "promo_line_share", "avg_discount_pct", "stockout_store_days"]
    out[fill0] = out[fill0].fillna(0)

    attrs = con.sql(f"""
        WITH top AS (
            SELECT sku_id, ROW_NUMBER() OVER (ORDER BY SUM(total_value) DESC) AS revenue_rank
            FROM sales GROUP BY 1 ORDER BY revenue_rank LIMIT {top_n}
        )
        SELECT top.*, sku_name, category, subcategory, brand, unit_price, cost_price
        FROM top JOIN skus USING (sku_id)
    """).df()
    out = out.merge(attrs, on="sku_id").sort_values(["revenue_rank", "week_end"]).reset_index(drop=True)

    g = out.groupby("sku_id")["units"]
    for lag in [1, 4, 52]:
        out[f"units_lag_{lag}"] = g.shift(lag)
    out["units_roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    return out


def inventory_features(con, snapshot) -> pd.DataFrame:
    """Stock ratios and sales velocity per store x SKU. `label_*` columns are for evaluation only."""
    S = f"DATE '{_date(snapshot)}'"
    f = con.sql(f"""
        WITH vel AS (
            SELECT store_id, sku_id,
                SUM(quantity) FILTER (WHERE date >= {S} - 14)  AS units_14,
                SUM(quantity) FILTER (WHERE date >= {S} - 28)  AS units_28,
                SUM(quantity) FILTER (WHERE date >= {S} - 90)  AS units_90,
                SUM(quantity) FILTER (WHERE date >= {S} - 365) AS units_365,
                MAX(date) AS last_sale_date
            FROM sales WHERE date < {S} GROUP BY 1, 2
        ),
        pairs AS (
            SELECT sku_id, flag, UNNEST(string_split(affected_stores, ';')) AS store_id FROM flags
        )
        SELECT i.*, k.category, k.cost_price, st.store_type,
               v.units_14, v.units_28, v.units_90, v.units_365, v.last_sale_date,
               (so.sku_id IS NOT NULL)::INT AS label_stockout_risk,
               (sm.sku_id IS NOT NULL)::INT AS label_slow_mover
        FROM inventory i
        JOIN skus k USING (sku_id)
        JOIN stores st USING (store_id)
        LEFT JOIN vel v USING (store_id, sku_id)
        LEFT JOIN pairs so ON so.flag = 'STOCKOUT_RISK' AND so.sku_id = i.sku_id AND so.store_id = i.store_id
        LEFT JOIN pairs sm ON sm.flag = 'SLOW_MOVER'    AND sm.sku_id = i.sku_id AND sm.store_id = i.store_id
        ORDER BY store_id, sku_id
    """).df()

    snap = pd.Timestamp(snapshot)
    units = ["units_14", "units_28", "units_90", "units_365"]
    f[units] = f[units].fillna(0)
    f["avg_daily_units_90"] = f["units_90"] / 90
    f["avg_daily_units_365"] = f["units_365"] / 365
    f["days_of_cover"] = f["stock_on_hand"] / f["avg_daily_units_90"].replace(0, np.nan)
    f["stock_to_reorder_point"] = f["stock_on_hand"] / f["reorder_point"]
    f["stock_to_safety_stock"] = f["stock_on_hand"] / f["safety_stock"].replace(0, np.nan)
    f["below_reorder_point"] = (f["stock_on_hand"] <= f["reorder_point"]).astype(int)
    f["out_of_stock"] = (f["stock_on_hand"] == 0).astype(int)
    f["stock_value"] = f["stock_on_hand"] * f["cost_price"]
    f["days_since_restock"] = (snap - pd.to_datetime(f["last_restock_date"])).dt.days
    f["days_since_last_sale"] = (snap - pd.to_datetime(f["last_sale_date"])).dt.days
    f["velocity_ratio_28_365"] = (f["units_28"] / 28) / f["avg_daily_units_365"].replace(0, np.nan)
    f["sudden_zero_sales"] = ((f["units_14"] == 0) & (f["avg_daily_units_365"] >= 0.5)).astype(int)
    return f.drop(columns=["last_sale_date"])
