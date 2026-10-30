import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from utils import FIG, RISK_COLORS, SEGMENTS, load_csv, page, pkr  # noqa: E402

page("Churn risk", "Probability that a customer's spend next quarter (Q1 2026) falls more than 30% below the same quarter last year (Step 4).")

churn = load_csv("customer_churn.csv")
threshold = st.sidebar.slider("Flag customers with churn probability above", 0.05, 0.95, 0.70, 0.05)
flagged = churn[churn["churn_prob"] >= threshold]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers flagged", f"{len(flagged):,}", f"{len(flagged) / len(churn):.1%} of all", delta_color="off")
c2.metric("Their spend last year", pkr(flagged["spend_365"].sum()), help="Revenue at risk if they cut back")
c3.metric("Average churn probability", f"{churn['churn_prob'].mean():.1%}", help="All customers")
c4.metric("High-risk band (> 0.7)", f"{(churn['risk_band'] == 'High').sum():,}")

left, right = st.columns(2)
bands = churn["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).rename_axis("risk_band").reset_index(name="customers")
fig = px.bar(bands, x="risk_band", y="customers", color="risk_band", color_discrete_map=RISK_COLORS, text_auto=",",
             title="Customers per risk band (customers)")
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="customers")
left.plotly_chart(fig, width="stretch")

by_seg = churn.groupby("Segment").agg(avg_prob=("churn_prob", "mean"), flagged=("churn_prob", lambda p: (p >= threshold).sum())).reindex(SEGMENTS)
fig = px.bar(by_seg.reset_index(), x="Segment", y=by_seg["avg_prob"].values * 100, text_auto=".0f",
             title="Average churn probability by segment (%)")
fig.update_layout(xaxis_title="", yaxis_title="%")
right.plotly_chart(fig, width="stretch")

left, right = st.columns([3, 2])
left.subheader("What drives churn")
left.image(str(FIG / "04c_churn_shap.png"), caption="SHAP values: features pushing risk up (right) or down (left)")
right.subheader("Flagged customers by segment")
right.dataframe(by_seg.assign(avg_prob=lambda x: (x["avg_prob"] * 100).round(1)).rename(
    columns={"avg_prob": "avg probability %", "flagged": "flagged"}), width="stretch")
right.caption("Medium/high-risk Champions and Loyal customers are the first people to contact: they spend the most.")

st.subheader(f"Top 50 highest-risk customers (threshold {threshold:.2f})")
top = flagged.sort_values("churn_prob", ascending=False).head(50)
st.dataframe(top.style.format({"churn_prob": "{:.1%}", "spend_365": "PKR {:,.0f}", "avg_days_between": "{:.1f}"}),
             width="stretch", hide_index=True)
st.download_button(f"Download all {len(flagged):,} flagged customers (CSV)", flagged.to_csv(index=False),
                   "churn_flagged.csv", "text/csv")
