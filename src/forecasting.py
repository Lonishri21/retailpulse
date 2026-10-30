"""Demand forecasting: lost-sales estimate, baselines, Prophet and LSTM."""
import logging

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)

H, F = 56, 7            # LSTM: days of history in, days forecast out
MONTHS = [1, 2, 11, 12]  # months with a step change in demand


def calendar(start="2022-01-01", end="2026-12-31") -> pd.DataFrame:
    """Known-in-advance calendar features per day."""
    c = pd.DataFrame(index=pd.date_range(start, end, name="date"))
    for m in MONTHS:
        c[f"m{m}"] = (c.index.month == m).astype(float)
    c["dow_sin"] = np.sin(2 * np.pi * c.index.dayofweek / 7)
    c["dow_cos"] = np.cos(2 * np.pi * c.index.dayofweek / 7)
    return c


def promo_calendar(con, index) -> pd.Series:
    """Store-wide discount % per day (0 when no store-wide promo)."""
    s = pd.Series(0.0, index=index)
    for r in con.sql("SELECT start_date, end_date, discount_pct FROM promos WHERE target_type = 'All'").df().itertuples():
        s.loc[r.start_date:r.end_date] = np.maximum(s.loc[r.start_date:r.end_date], r.discount_pct)
    return s


def lost_units(con, by_sku: bool = False) -> pd.DataFrame:
    """Estimated units lost to known stockouts.

    Each store x SKU in a stockout window is assumed to have sold its average daily units
    from the 90 days before the window.
    """
    key = "sku_id, " if by_sku else ""
    df = con.sql(f"""
        WITH pairs AS (
            SELECT sku_id, UNNEST(string_split(affected_stores, ';')) AS store_id,
                   CAST(window_start AS DATE) AS ws, CAST(window_end AS DATE) AS we
            FROM flags WHERE flag = 'STOCKOUT_RISK'
        ), rate AS (
            SELECT p.sku_id, p.store_id, p.ws, p.we, COALESCE(SUM(s.quantity), 0) / 90.0 AS r
            FROM pairs p LEFT JOIN sales s
              ON s.sku_id = p.sku_id AND s.store_id = p.store_id AND s.date >= p.ws - 90 AND s.date < p.ws
            GROUP BY ALL
        ), days AS (
            SELECT sku_id, r, UNNEST(generate_series(ws, we, INTERVAL 1 DAY)) AS d FROM rate
        )
        SELECT {key}CAST(d AS DATE) AS date, SUM(r) AS lost_units FROM days GROUP BY ALL ORDER BY ALL
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- daily models
def rolling_origin(forecast_fn, y: pd.Series, start, weeks: int = 8) -> pd.Series:
    """Forecast 7 days from each weekly origin, using only data before the origin."""
    parts = []
    for o in pd.date_range(start, periods=weeks, freq="7D"):
        parts.append(forecast_fn(y.loc[: o - pd.Timedelta(days=1)], pd.date_range(o, periods=F)))
    return pd.concat(parts)


def naive_last_week(history: pd.Series, days) -> pd.Series:
    return pd.Series(history.iloc[-7:].to_numpy(), index=days)


def last_year_with_growth(history: pd.Series, days) -> pd.Series:
    """Same weekday last year x growth of the last 28 days vs the same 28 days last year."""
    growth = history.iloc[-28:].sum() / history.iloc[-28 - 364:-364].sum()
    return pd.Series(history.reindex(days - pd.Timedelta(days=364)).to_numpy() * growth, index=days)


def prophet_model(history: pd.Series, regressors: pd.DataFrame, interval_width: float = 0.8):
    """Fit Prophet (multiplicative yearly + weekly seasonality) with the given regressors."""
    from prophet import Prophet
    m = Prophet(yearly_seasonality=10, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", interval_width=interval_width)
    for col in regressors.columns:
        m.add_regressor(col)
    df = pd.DataFrame({"ds": history.index, "y": history.to_numpy()}).join(regressors, on="ds")
    return m.fit(df)


def prophet_predict(model, days, regressors: pd.DataFrame) -> pd.DataFrame:
    fut = pd.DataFrame({"ds": days}).join(regressors, on="ds")
    return model.predict(fut).set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]


# --------------------------------------------------------------------------- LSTM
def lstm_inputs(origin, y: pd.Series, extra: pd.DataFrame):
    """Inputs for one forecast origin, scaled by the recent level.

    history: last H days of (units / level, discount)
    future:  next F days of (same day last year, relative to last year's level), calendar, discount
    """
    hist = y.loc[origin - pd.Timedelta(days=H): origin - pd.Timedelta(days=1)]
    level = hist.iloc[-28:].mean()
    level_ly = y.loc[origin - pd.Timedelta(days=364 + 28): origin - pd.Timedelta(days=365)].mean()
    days = pd.date_range(origin, periods=F)
    xh = np.column_stack([hist.to_numpy() / level, extra.loc[hist.index, "disc"].to_numpy()])
    xf = np.column_stack([y.reindex(days - pd.Timedelta(days=364)).to_numpy() / level_ly,
                          extra.loc[days].to_numpy()])
    return xh, xf, level


def lstm_dataset(origins, y: pd.Series, extra: pd.DataFrame, with_target: bool = True):
    xh, xf, lv, tg = [], [], [], []
    for o in origins:
        a, b, level = lstm_inputs(o, y, extra)
        xh.append(a); xf.append(b); lv.append(level)
        if with_target:
            tg.append(y.reindex(pd.date_range(o, periods=F)).to_numpy() / level)
    out = [np.array(xh, "float32"), np.array(xf, "float32"), np.array(lv)]
    return out + [np.array(tg, "float32")] if with_target else out


def train_lstm(y: pd.Series, extra: pd.DataFrame, last_origin, seed: int = 42):
    """Train on every daily origin whose 7 target days end before `last_origin` + 7."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    first = y.index[0] + pd.Timedelta(days=364 + 28 + 1)
    xh, xf, _, tg = lstm_dataset(pd.date_range(first, last_origin), y, extra)

    ih, i_f = tf.keras.Input((H, xh.shape[2])), tf.keras.Input((F, xf.shape[2]))
    z = tf.keras.layers.Concatenate()([tf.keras.layers.LSTM(32)(ih), tf.keras.layers.Flatten()(i_f)])
    z = tf.keras.layers.Dense(64, activation="relu")(z)
    model = tf.keras.Model([ih, i_f], tf.keras.layers.Dense(F)(z))
    model.compile(tf.keras.optimizers.Adam(1e-3), "mse")
    model.fit([xh, xf], tg, epochs=300, batch_size=32, validation_split=0.15, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])
    return model


def lstm_forecast(model, y: pd.Series, extra: pd.DataFrame, origins) -> pd.Series:
    xh, xf, lv = lstm_dataset(origins, y, extra, with_target=False)
    pred = model.predict([xh, xf], verbose=0) * lv[:, None]
    days = np.concatenate([pd.date_range(o, periods=F) for o in origins])
    return pd.Series(pred.ravel(), index=pd.DatetimeIndex(days))
