import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from utils import FIG, load_csv, page  # noqa: E402

page("Model monitoring", "Data drift (2025 vs 2024), forecast error over time, and the experiment log (Step 7).")

drift = load_csv("drift_report.csv")
bt = load_csv("forecast_backtest.csv", parse_dates=["date"])
cmp = load_csv("forecast_comparison.csv", index_col=0)

wk = bt.assign(week=bt["date"] + pd.to_timedelta(6 - bt["date"].dt.dayofweek, unit="D")).groupby("week")[["Hybrid", "actual"]].sum()
wk["rolling_wape"] = (wk["actual"] - wk["Hybrid"]).abs().rolling(4).sum() / wk["actual"].rolling(4).sum() * 100
thresh = cmp.loc["Hybrid", "test weekly WAPE %"] + 5
key = drift[drift["feature"].isin(["units", "receipts", "customers", "spend_90", "visits_90"]) & (drift["PSI"] > 0.25)]

c1, c2, c3 = st.columns(3)
c1.metric("Max rolling 4-week error", f"{wk['rolling_wape'].max():.1f}%", f"threshold {thresh:.1f}%", delta_color="off")
c2.metric("Features with significant drift", f"{(drift['status'] == 'Significant').sum()} of {len(drift)}")
c3.metric("Retraining decision", "Retrain" if len(key) or wk["rolling_wape"].max() > thresh else "No action",
          help="Rule: retrain if rolling 4-week WAPE > test WAPE + 5 points, or a key feature has PSI > 0.25")

left, right = st.columns(2)
fig = px.bar(drift.sort_values("PSI"), x="PSI", y="feature", color="status", orientation="h", facet_row="group",
             color_discrete_map={"Stable": "#4C9F70", "Moderate": "#E0A030", "Significant": "#C8553D"},
             title="Drift by feature, 2025 vs 2024 (PSI)")
fig.add_vline(x=0.1, line_dash="dot", line_color="grey"); fig.add_vline(x=0.25, line_dash="dash", line_color="grey")
fig.update_yaxes(matches=None, title=""); fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(height=520, legend_title="")
left.plotly_chart(fig, width="stretch")

fig = go.Figure()
fig.add_trace(go.Scatter(x=wk.index, y=wk["rolling_wape"], mode="lines+markers", name="rolling 4-week WAPE"))
fig.add_hline(y=thresh, line_dash="dash", line_color="#C8553D", annotation_text=f"retrain threshold {thresh:.1f}%")
fig.update_layout(title="Hybrid forecast: rolling 4-week error (%)", yaxis_title="%", xaxis_title="", yaxis_range=[0, thresh + 2], height=520)
right.plotly_chart(fig, width="stretch")

st.markdown("""
**Reading the drift:** daily sales features drift mainly because the business **grew** (~5% more units) and ran **far fewer store-wide promotions** in 2025
(revenue +12.5%). Customer behaviour (visit frequency, basket size, recency) is stable. The models are not broken, but they need **regular retraining**.

**Retraining rule:** retrain forecasts if the rolling 4-week WAPE rises 5 points above the test WAPE, or a key sales feature has PSI > 0.25;
re-score churn monthly; recalculate safety stocks whenever forecasts are retrained.
""")

with st.expander("Drift report (all features)"):
    st.dataframe(drift.style.format({"PSI": "{:.3f}", "KS_stat": "{:.3f}", "KS_p": "{:.4f}", "ref_mean": "{:,.2f}",
                                     "cur_mean": "{:,.2f}", "mean_change_%": "{:+.1f}"}), width="stretch", hide_index=True)

st.subheader("Experiment log (MLflow)")
runs = load_csv("mlflow_runs.csv")
churn_cols = ["runName", "val_ROC_AUC", "val_PR_AUC", "test_ROC_AUC", "test_PR_AUC"]
fc_cols = ["runName", "val_weekly_WAPE_pct", "test_weekly_WAPE_pct", "test_vs_demand_weekly_WAPE_pct", "test_daily_WAPE_pct"]
left, right = st.columns(2)
left.caption("Churn experiment")
left.dataframe(runs[runs["experiment"] == "retailpulse-churn"][[c for c in churn_cols if c in runs]].round(3), width="stretch", hide_index=True)
right.caption("Forecasting experiment")
right.dataframe(runs[runs["experiment"] == "retailpulse-forecasting"][[c for c in fc_cols if c in runs]].round(2), width="stretch", hide_index=True)
st.image(str(FIG / "07a_optuna_churn.png"), caption="Optuna tuning history for the churn model")
