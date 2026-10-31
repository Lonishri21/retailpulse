"""RFM scores and K-Means customer segments."""
import numpy as np
import pandas as pd

KMEANS_FEATURES = ["recency_days", "frequency", "monetary"]
SEGMENT_ORDER = ["Champions", "Loyal", "Regulars", "Occasional", "Cooling Off", "Slipping Away"]

SEGMENT_ACTIONS = {
    "Champions":       ("Shop almost daily, highest spend", "VIP perks, early access to new products, referral rewards"),
    "Loyal":           ("Shop every 2 days, high spend", "Bundles and cross-sell to move them up to Champions"),
    "Regulars":        ("Shop every 3 days, mid spend", "Personalised offers in categories they already buy"),
    "Occasional":      ("Shop every 4-5 days, lowest spend", "Visit-frequency incentives (e.g. points for a 2nd visit in a week)"),
    "Cooling Off":     ("Regular shoppers who are a few days late", "Friendly reminder with a personalised offer"),
    "Slipping Away":   ("Infrequent shoppers, longest since last visit", "Win-back message with a small, time-limited offer"),
}


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, M scores (1-5), RFM_Score and RFM_Segment."""
    out = df.copy()
    out["R"] = pd.qcut(out["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_Score"] = out[["R", "F", "M"]].sum(axis=1)
    out["RFM_Segment"] = [rfm_segment(r, f, m) for r, f, m in zip(out["R"], out["F"], out["M"])]
    return out


def rfm_segment(r: int, f: int, m: int) -> str:
    """Rule-based RFM segment name."""
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


def make_kmeans(k: int, random_state: int = 42):
    """log1p -> standardise -> K-Means pipeline."""
    from sklearn.cluster import KMeans  # imported here because the dashboard doesn't install sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    return make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=random_state),
    )


def name_clusters(df: pd.DataFrame, labels) -> dict:
    """The two clusters with the longest recency are Slipping Away and Cooling Off; the rest are named by frequency."""
    prof = df.assign(cluster=labels).groupby("cluster")[["recency_days", "frequency"]].mean()
    if len(prof) != len(SEGMENT_ORDER):
        raise ValueError(f"name_clusters expects {len(SEGMENT_ORDER)} clusters, got {len(prof)}")
    late = prof["recency_days"].nlargest(2).index
    rest = prof.drop(index=late).sort_values("frequency", ascending=False).index
    return {late[0]: "Slipping Away", late[1]: "Cooling Off", **dict(zip(rest, SEGMENT_ORDER[:4]))}
