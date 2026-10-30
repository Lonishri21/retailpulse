import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from utils import load_csv, page  # noqa: E402

page("Demand forecast", "Units sold: baselines vs Prophet vs LSTM vs hybrid, the 30-day forecast for January 2026, and weekly forecasts for the top 20 products (Step 5).")

daily = load_csv("daily_sales_features.csv", parse_dates=["date"])
f30 = load_csv("forecast_30d.csv", parse_dates=["date"])
cmp = load_csv("forecast_comparison.csv", index_col=0)
bt = load_csv("forecast_backtest.csv", parse_dates=["date"])

c1, c2, c3 = st.columns(3)
c1.metric("January 2026 forecast", f"{f30['forecast'].sum():,.0f} units",
          f"80% range {f30['lower_80'].sum() / 1e3:,.0f}k–{f30['upper_80'].sum() / 1e3:,.0f}k", delta_color="off")
ly = daily.set_index("date")["units"].reindex(f30["date"] - pd.Timedelta(days=364)).sum()
c2.metric("vs same days of 2025", f"{f30['forecast'].sum() / ly - 1:+.1%}")
c3.metric("Hybrid weekly error (test)", f"{cmp.loc['Hybrid', 'test weekly WAPE %']:.1f}%")

hist = daily[daily["date"] >= "2025-09-01"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["units"], name="recorded units", line=dict(color="black", width=1)))
fig.add_trace(go.Scatter(x=pd.concat([f30["date"], f30["date"][::-1]]), y=pd.concat([f30["upper_80"], f30["lower_80"][::-1]]),
                         fill="toself", fillcolor="rgba(200,85,61,.2)", line=dict(width=0), name="80% range"))
fig.add_trace(go.Scatter(x=f30["date"], y=f30["forecast"], name="hybrid forecast", line=dict(color="#C8553D", width=2)))
fig.update_layout(title="Daily units: recent history and 30-day forecast (units)", yaxis_title="units", xaxis_title="")
st.plotly_chart(fig, width="stretch")

left, right = st.columns([2, 3])
left.subheader("Model comparison")
left.dataframe(cmp[["val weekly WAPE %", "test weekly WAPE %", "test weekly WAPE % vs est. demand"]].style.format("{:.2f}"), width="stretch")
left.caption("WAPE = total absolute error ÷ total units (lower is better). 'Est. demand' adds units lost to stockouts.")
test = bt[bt["split"] == "test"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=test["date"], y=test["actual"], name="recorded units", line=dict(color="black", width=2)))
fig.add_trace(go.Scatter(x=test["date"], y=test["estimated_demand"], name="estimated demand", line=dict(color="black", dash="dot")))
for col, color in [("Prophet", "#2E5EAA"), ("LSTM", "#4C9F70"), ("Hybrid", "#C8553D")]:
    fig.add_trace(go.Scatter(x=test["date"], y=test[col], name=col, line=dict(color=color, width=1.3)))
fig.update_layout(title="Test weeks: 7-day-ahead forecasts vs actual (units)", yaxis_title="units", xaxis_title="", legend_font_size=9)
right.plotly_chart(fig, width="stretch")

st.subheader("Top 20 products: weekly forecast")
sku_w = load_csv("sku_weekly.csv", parse_dates=["week_end"])
sku_f = load_csv("sku_forecasts.csv", parse_dates=["week_end"])
names = sku_w.drop_duplicates("sku_id").sort_values("revenue_rank").set_index("sku_id")["sku_name"]
sku = st.selectbox("Product", names.index, format_func=lambda s: f"{s} — {names[s]}")
h, f = sku_w[(sku_w["sku_id"] == sku) & (sku_w["week_end"] >= "2025-01-01")], sku_f[sku_f["sku_id"] == sku]
fig = go.Figure()
fig.add_trace(go.Scatter(x=h["week_end"], y=h["units"], name="recorded weekly units", line=dict(color="black")))
fig.add_trace(go.Scatter(x=f["week_end"], y=f["forecast_units"], name="forecast", mode="lines+markers", line=dict(color="#C8553D"),
                         error_y=dict(type="data", array=f["error_std"], visible=True)))
fig.update_layout(title=f"{sku}: weekly units, 2025 and the next 5 weeks (units; bars = ±1 typical weekly error)",
                  yaxis_title="units", xaxis_title="")
st.plotly_chart(fig, width="stretch")
st.caption(f"Backtest error for this product: {f['backtest_wape_pct'].iloc[0]:.1f}% WAPE (1–8 weeks ahead, vs estimated demand). "
           "Dips in late 2025 are injected stockouts, not lower demand.")
