"""RetailPulse dashboard: home page. Run from the project root: streamlit run app/Home.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st  # noqa: E402
from utils import load_csv, page, pkr  # noqa: E402

page("RetailPulse", "AI-powered customer analytics and demand forecasting for a 30-store retail chain (synthetic data, 2022–2025, PKR).")

daily = load_csv("daily_sales_features.csv", parse_dates=["date"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue (2022–2025)", pkr(daily["revenue"].sum()))
c2.metric("Units sold", f"{daily['units'].sum() / 1e6:,.1f}m")
c3.metric("Average basket", pkr(daily["revenue"].sum() / daily["receipts"].sum()))
c4.metric("Gross margin", f"{daily['gross_profit'].sum() / daily['revenue'].sum():.1%}")

st.markdown("""
RetailPulse turns four years of sales into decisions: **who** the customers are and who is likely to cut back,
**how much** each product will sell in the coming weeks, and **how much stock** to hold so shelves stay full without tying up cash.
Every number on these pages comes from models trained and tested in the project notebooks; the dashboard only reads their saved results.
""")

st.subheader("Pipeline")
st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2F8", color="#2E5EAA", fontname="Helvetica", fontsize=11];
  raw [label="Sales data\\n10M lines"]; clean [label="Cleaning +\\nfeatures"]; seg [label="Segments\\n(RFM + K-Means)"];
  churn [label="Churn model\\n(XGBoost)"]; fc [label="Forecasts\\n(Prophet + LSTM)"]; inv [label="Inventory\\n(SS, ROP, EOQ)"];
  mlops [label="MLflow, Optuna,\\ndrift checks"]; app [label="Dashboard", fillcolor="#2E5EAA", fontcolor="white"];
  raw -> clean -> seg -> churn -> app; clean -> fc -> inv -> app; churn -> mlops; fc -> mlops; mlops -> app;
}
""")

st.subheader("Key results")
runs = load_csv("mlflow_runs.csv").set_index("runName")
fcmp = load_csv("forecast_comparison.csv", index_col=0)
churn = load_csv("customer_churn.csv")
sim = load_csv("inventory_simulation.csv")
cost = sim.groupby("policy")["total_cost"].sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customer segments", "5", help="K-Means on recency, frequency and spend")
k2.metric("Churn model ROC-AUC (test)", f"{runs.loc['xgboost_baseline', 'test_ROC_AUC']:.2f}",
          help="Out-of-time test quarter (Oct–Dec 2025)")
k3.metric("Weekly forecast error (test)", f"{fcmp.loc['Hybrid', 'test weekly WAPE %']:.1f}%", help="WAPE of the hybrid forecast")
k4.metric("Inventory cost vs naive policy", f"{cost['Recommended (seasonal)'] / cost['Naive'] - 1:+.0%}",
          help="Simulated on 2025 demand, same fill rate")
st.markdown(f"""
- **Customers** differ in how *often* they shop, not how much per visit; the *Slipping Away* segment ({(load_csv('customer_segments.csv')['Segment'] == 'Slipping Away').mean():.0%} of customers) is overdue for a visit.
- **{(churn['risk_band'] == 'High').sum()} customers are high risk** of cutting spend by more than 30% next quarter.
- **Demand** grows ~5% a year with a sharp November–December peak; promotions cut revenue but barely change units.
- **Inventory:** demand-based reorder points give the same service as a naive policy with 37% less stock.

Use the pages in the sidebar to explore each area.
""")
