import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from utils import SEGMENT_COLORS, SEGMENTS, load_csv, page, pkr  # noqa: E402
from src.segmentation import SEGMENT_ACTIONS  # noqa: E402

page("Customer segments", "Six segments from K-Means on recency, frequency and spend (Step 3).")

seg = load_csv("customer_segments.csv")
churn = load_csv("customer_churn.csv")[["customer_id", "churn_prob", "risk_band"]]
chosen = st.sidebar.multiselect("Segments", SEGMENTS, default=SEGMENTS)
s = seg[seg["Segment"].isin(chosen)]

prof = (seg.groupby("Segment").agg(customers=("customer_id", "size"), revenue=("monetary", "sum"),
                                   visits_4y=("frequency", "mean"), days_between_visits=("avg_days_between", "mean"),
                                   days_since_last_visit=("recency_days", "mean"), basket_value=("aov", "mean"))
        .reindex(SEGMENTS))
prof["share_of_customers_%"] = prof["customers"] / prof["customers"].sum() * 100
prof["share_of_revenue_%"] = prof["revenue"] / prof["revenue"].sum() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Customers selected", f"{len(s):,}")
c2.metric("Their revenue (2022–2025)", pkr(s["monetary"].sum()))
c3.metric("Share of total revenue", f"{s['monetary'].sum() / seg['monetary'].sum():.0%}")

left, right = st.columns(2)
share = prof.loc[chosen, ["share_of_customers_%", "share_of_revenue_%"]].reset_index().melt(
    id_vars="Segment", var_name="measure", value_name="share")
share["measure"] = share["measure"].map({"share_of_customers_%": "customers", "share_of_revenue_%": "revenue"})
fig = px.bar(share, x="Segment", y="share", color="measure", barmode="group", text_auto=".0f",
             title="Share of customers vs share of revenue (%)")
fig.update_layout(yaxis_title="%", xaxis_title="", legend_title="")
left.plotly_chart(fig, width="stretch")

fig = px.scatter(s.sample(min(len(s), 4000), random_state=42), x="frequency", y="monetary", color="Segment",
                 color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENTS}, opacity=.6,
                 hover_data=["customer_id", "recency_days", "loyalty_segment"],
                 title="Visits vs spend per customer, 2022–2025 (baskets, PKR)")
fig.update_traces(marker_size=4)
fig.update_layout(xaxis_title="baskets", yaxis_title="PKR", legend_title="")
right.plotly_chart(fig, width="stretch")

st.subheader("Segment profiles and actions")
actions = pd.DataFrame(SEGMENT_ACTIONS, index=["Who they are", "Recommended action"]).T
table = prof[["customers", "share_of_customers_%", "share_of_revenue_%", "visits_4y", "days_between_visits",
              "days_since_last_visit", "basket_value"]].join(actions).loc[chosen]
st.dataframe(table.style.format({"customers": "{:,.0f}", "share_of_customers_%": "{:.1f}", "share_of_revenue_%": "{:.1f}",
                                 "visits_4y": "{:,.0f}", "days_between_visits": "{:.1f}", "days_since_last_visit": "{:.1f}",
                                 "basket_value": "PKR {:,.0f}"}), width="stretch")

st.subheader("Customer lookup")
cid = st.text_input("Customer ID", placeholder="e.g. CUST00001").strip().upper()
if cid:
    row = seg[seg["customer_id"] == cid].merge(churn, on="customer_id")
    if row.empty:
        st.warning(f"No customer with ID {cid}.")
    else:
        r = row.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Segment", r["Segment"])
        b.metric("Loyalty tier", r["loyalty_segment"])
        c.metric("Churn risk (next quarter)", f"{r['churn_prob']:.0%} ({r['risk_band']})")
        d.metric("Spend 2022–2025", pkr(r["monetary"]))
        st.write(f"**Recommended action:** {SEGMENT_ACTIONS[r['Segment']][1]}")
        st.dataframe(row.drop(columns=["Cluster"]), width="stretch", hide_index=True)
