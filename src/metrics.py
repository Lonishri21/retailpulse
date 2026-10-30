"""Forecast accuracy metrics."""
import numpy as np


def forecast_metrics(y, yhat) -> dict:
    """MAE, RMSE, WAPE % (headline) and MAPE %."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "WAPE_%": np.sum(np.abs(err)) / np.sum(np.abs(y)) * 100,
        "MAPE_%": np.mean(np.abs(err[y > 0] / y[y > 0])) * 100,
    }
