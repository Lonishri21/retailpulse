"""Data drift: Population Stability Index (PSI) and Kolmogorov-Smirnov test."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins: int = 10) -> float:
    """PSI with quantile bins taken from `expected`."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_status(value: float) -> str:
    return "Stable" if value < 0.1 else "Moderate" if value < 0.25 else "Significant"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, columns) -> pd.DataFrame:
    """PSI, KS test and mean change for each column."""
    rows = []
    for c in columns:
        r, k = ref[c].dropna(), cur[c].dropna()
        p, ks = psi(r, k), ks_2samp(r, k)
        rows.append({"feature": c, "PSI": p, "KS_stat": ks.statistic, "KS_p": ks.pvalue,
                     "ref_mean": r.mean(), "cur_mean": k.mean(), "mean_change_%": (k.mean() / r.mean() - 1) * 100 if r.mean() else np.nan,
                     "status": drift_status(p)})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
