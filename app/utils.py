"""Shared loaders and formatting for the dashboard pages."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, RAW = ROOT / "data" / "processed", ROOT / "reports" / "figures", ROOT / "retail_clean_dataset"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEGMENTS = ["Champions", "Loyal", "Regulars", "Occasional", "Slipping Away"]
SEGMENT_COLORS = dict(zip(SEGMENTS, ["#2E5EAA", "#4C9F70", "#E0A030", "#C8553D", "#7A6FA8"]))
RISK_COLORS = {"Low": "#4C9F70", "Medium": "#E0A030", "High": "#C8553D"}


@st.cache_data
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from data/processed."""
    return pd.read_csv(DATA / name, **kwargs)


@st.cache_data
def load_master(name: str) -> pd.DataFrame:
    """Read a master table (stores, SKUs, customers)."""
    return pd.read_csv(RAW / f"{name}.csv")


def pkr(x: float) -> str:
    """Compact PKR amount, e.g. PKR 10.9bn, PKR 4.2m, PKR 2,106."""
    for div, unit in [(1e9, "bn"), (1e6, "m")]:
        if abs(x) >= div:
            return f"PKR {x / div:,.1f}{unit}"
    return f"PKR {x:,.0f}"


def page(title: str, intro: str = "") -> None:
    st.set_page_config(page_title=f"{title} | RetailPulse", page_icon="📊", layout="wide")
    st.title(title)
    if intro:
        st.caption(intro)
