import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.stats import norm  # noqa: E402
from utils import FIG, load_csv, page, pkr  # noqa: E402

page("Inventory", "Safety stock, reorder point and order quantity for the top 20 products, January 2026 (Step 6). "
     "Assumes a 1-week lead time, PKR 5,000 per order and holding cost of 20% of cost price per year.")

rec = load_csv("inventory_recommendations.csv")
level = st.sidebar.slider("Service level (chance of not running out in a week)", 0.80, 0.99, 0.95, 0.01)
abc = st.sidebar.multiselect("ABC class (within top 20)", ["A", "B", "C"], default=["A", "B", "C"])

# Recomputed live from the saved forecast and error: nothing is retrained
z = norm.ppf(level)
r = rec[rec["abc_top20"].isin(abc)].copy()
r["safety_stock"] = z * r["sigma"]
r["reorder_point"] = r["forecast_weekly"] + r["safety_stock"]
r["status"] = np.select([r["stock_position"] <= r["reorder_point"], r["stock_position"] > r["reorder_point"] + 2 * r["eoq"]],
                        ["Reorder now", "Overstock"], "OK")

c1, c2, c3, c4 = st.columns(4)
c1.metric("z-score", f"{z:.2f}")
c2.metric("Total safety stock value", pkr((r["safety_stock"] * r["cost_price"]).sum()))
c3.metric("Products to reorder now", int((r["status"] == "Reorder now").sum()))
c4.metric("Overstocked products", int((r["status"] == "Overstock").sum()))

cols = ["sku_name", "abc_top20", "forecast_weekly", "sigma", "safety_stock", "reorder_point", "eoq", "stock_position", "status"]
st.dataframe(r.set_index("sku_id")[cols].style.format({"forecast_weekly": "{:,.0f}", "sigma": "{:,.0f}", "safety_stock": "{:,.0f}",
                                                       "reorder_point": "{:,.0f}", "eoq": "{:,.0f}", "stock_position": "{:,.0f}"}),
             width="stretch")
st.caption("Units per week for the whole chain. σ = typical weekly forecast error. Stock position = simulated stock + on order at the end of 2025.")

left, right = st.columns(2)
r["safety_stock_%_of_demand"] = r["safety_stock"] / r["forecast_weekly"] * 100
fig = px.bar(r.sort_values("safety_stock_%_of_demand"), x="safety_stock_%_of_demand", y="sku_id", orientation="h",
             color="abc_top20", title=f"Safety stock at {level:.0%} service (% of weekly demand)")
fig.update_layout(xaxis_title="% of weekly demand", yaxis_title="", legend_title="ABC")
left.plotly_chart(fig, width="stretch")

sim = load_csv("inventory_simulation.csv")
tot = sim.groupby("policy").agg(fill=("fill_rate_%", "mean"), stock=("avg_inventory_value", "sum"), cost=("total_cost", "sum")).reset_index()
fig = px.bar(tot, x="policy", y=tot["cost"] / 1e6, color="policy", text=tot["cost"].map(pkr),
             title="Simulated 2025 holding + ordering cost, top 20 products (PKR m)")
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="PKR m")
right.plotly_chart(fig, width="stretch")
right.caption(f"Average fill rate: {', '.join(f'{p} {f:.1f}%' for p, f in zip(tot['policy'], tot['fill']))}.")

st.subheader("Simulated stock in 2025")
st.image(str(FIG / "06b_simulation.png"))
