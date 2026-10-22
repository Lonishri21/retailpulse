"""Customer segmentation helpers: rule-based RFM scores and K-Means cluster naming.

Shared by the segmentation notebook and the dashboard, so segments are assigned the same way everywhere.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

KMEANS_FEATURES = ["recency_days", "frequency", "monetary"]

# Business action for each K-Means segment (used in the report and the dashboard)
SEGMENT_ACTIONS = {
    "Champions":     ("Shop almost daily, highest spend", "VIP perks, early access to new products, referral rewards"),
    "Loyal":         ("Shop every 2 days, high spend", "Bundles and cross-sell to move them up to Champions"),
    "Regulars":      ("Shop every 3 days, mid spend", "Personalised offers in categories they already buy"),
    "Occasional":    ("Shop every 4-5 days, lowest spend", "Visit-frequency incentives (e.g. points for a 2nd visit in a week)"),
    "Slipping Away": ("Longer than usual since last visit", "Timely win-back message with a small, time-limited offer"),
}


# --------------------------------------------------------------------------- RFM rules
def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, M scores (1-5) and RFM_Score. Recent / frequent / high spend = 5.

    Recency and Frequency are ranked first because many customers share the same value
    and qcut fails on duplicate bin edges.
    """
    out = df.copy()
    out["R"] = pd.qcut(out["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_Score"] = out[["R", "F", "M"]].sum(axis=1)
    out["RFM_Segment"] = [rfm_segment(r, f, m) for r, f, m in zip(out["R"], out["F"], out["M"])]
    return out


def rfm_segment(r: int, f: int, m: int) -> str:
    """Classic rule-based RFM segment names."""
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and m >= 4:
        return "Can't Lose Them"
    if r <= 2:
        return "Lost / Dormant"
    return "Needs Attention"


# --------------------------------------------------------------------------- K-Means
def make_kmeans(k: int, random_state: int = 42):
    """log1p -> standardise -> K-Means. RFM values are skewed, so the log stops big spenders dominating."""
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=random_state),
    )


def name_clusters(df: pd.DataFrame, labels) -> dict:
    """Map cluster numbers to business names from their profiles (cluster numbers change between runs).

    The cluster with the longest average recency is "Slipping Away"; the rest are named by
    average frequency, highest first.
    """
    prof = df.assign(cluster=labels).groupby("cluster")[["recency_days", "frequency"]].mean()
    slipping = prof["recency_days"].idxmax()
    rest = prof.drop(index=slipping).sort_values("frequency", ascending=False).index
    names = ["Champions", "Loyal", "Regulars", "Occasional"]
    if len(rest) != len(names):
        raise ValueError(f"name_clusters expects {len(names) + 1} clusters, got {len(prof)}")
    return {slipping: "Slipping Away", **dict(zip(rest, names))}
