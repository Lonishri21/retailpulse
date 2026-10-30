import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from utils import load_csv, load_master, page, pkr  # noqa: E402

page("Sales overview", "Chain-wide sales trends, seasonality, stores, categories and products.")

daily = load_csv("daily_sales_features.csv", parse_dates=["date"])
monthly = load_csv("sales_monthly.csv", parse_dates=["month"]).merge(load_master("store_master"), on="store_id")

rng = st.sidebar.date_input("Date range", [daily["date"].min(), daily["date"].max()],
                            min_value=daily["date"].min(), max_value=daily["date"].max())
if len(rng) != 2:
    st.stop()
start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
cities = st.sidebar.multiselect("City (store breakdowns)", sorted(monthly["city"].unique()))

d = daily[daily["date"].between(start, end)]
m = monthly[monthly["month"].between(start.to_period("M").to_timestamp(), end)]
if cities:
    m = m[m["city"].isin(cities)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", pkr(d["revenue"].sum()))
c2.metric("Baskets", f"{d['receipts'].sum():,}")
c3.metric("Units sold", f"{d['units'].sum():,}")
c4.metric("Average basket", pkr(d["revenue"].sum() / max(d["receipts"].sum(), 1)))

measure = st.radio("Measure", ["units", "revenue"], horizontal=True,
                   help="Promotions cut revenue without changing units, so units show demand more clearly")
ma = st.slider("Moving average (days)", 7, 60, 28)
t = d.assign(trend=d[measure].rolling(ma, min_periods=1).mean())
label = "units" if measure == "units" else "PKR"
fig = px.line(t, x="date", y=[measure, "trend"], title=f"Daily {measure} ({label}) with {ma}-day moving average")
fig.update_layout(yaxis_title=label, xaxis_title="", legend_title="")
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
cat = m.groupby([m["month"], "category"], as_index=False)["revenue"].sum()
fig = px.bar(cat, x="month", y="revenue", color="category", title="Monthly revenue by category (PKR)")
fig.update_layout(yaxis_title="PKR", xaxis_title="", legend_title="", legend_font_size=9)
left.plotly_chart(fig, width="stretch")

hm = d.assign(weekday=d["date"].dt.day_name().str[:3], month=d["date"].dt.month_name().str[:3])
hm = hm.pivot_table(index="weekday", columns="month", values="units", aggfunc="mean")
hm = hm.reindex(index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                columns=[mo for mo in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if mo in hm.columns])
fig = px.imshow(hm, color_continuous_scale="Blues", aspect="auto", title="Average daily units by weekday and month (units)")
right.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
stores = m.groupby(["store_name", "store_type"], as_index=False)["revenue"].sum().nlargest(15, "revenue")
fig = px.bar(stores.sort_values("revenue"), x="revenue", y="store_name", color="store_type", orientation="h",
             title="Top 15 stores by revenue (PKR)")
fig.update_layout(xaxis_title="PKR", yaxis_title="", legend_title="")
left.plotly_chart(fig, width="stretch")

prod = load_csv("product_sales.csv").merge(load_master("sku_master")[["sku_id", "sku_name", "category"]], on="sku_id").head(10)
fig = px.bar(prod.sort_values("revenue"), x="revenue", y="sku_name", color="category", orientation="h",
             title="Top 10 products by revenue, 2022–2025 (PKR)")
fig.update_layout(xaxis_title="PKR", yaxis_title="", legend_title="")
right.plotly_chart(fig, width="stretch")

st.download_button("Download daily data (CSV)", d.to_csv(index=False), "sales_daily.csv", "text/csv")
