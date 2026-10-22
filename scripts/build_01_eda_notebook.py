"""Builds notebooks/01_eda.ipynb (the Step 1 data analysis notebook). Run: python scripts/build_01_eda_notebook.py"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# RetailPulse — Step 1: Exploratory Data Analysis

**Goal of this step:** understand the data before building any model.

1. What data do we have, and how do the tables connect?
2. Is the data trustworthy? (quality checks)
3. What patterns exist in sales, stores, products, customers, promotions, baskets and inventory?
4. Which **features** should the later models (forecasting, segmentation, churn, inventory) use?

Data: synthetic retail chain in Pakistan — 30 stores, 5,000 SKUs, 10,000 customers, ~10M sales line items, 2022–2025.
""")

md("## 0. Setup\nDuckDB lets us query the 800 MB sales file with SQL without loading it all into memory.")
code("""
import duckdb, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.ticker as mt, seaborn as sns
from pathlib import Path
from IPython.display import display, Markdown

pd.set_option('display.float_format', '{:,.2f}'.format)
pd.set_option('display.width', 160)
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({'figure.figsize': (11, 4.5), 'figure.dpi': 110, 'axes.titleweight': 'bold'})

ROOT = Path('..')
MAIN = ROOT / 'retail_clean_dataset'          # main dataset used in this notebook
ALT  = ROOT / 'retail_contaminated_dataset'   # second version, compared in section 10
FIG = ROOT / 'reports' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'data' / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
def q(sql):  return con.sql(sql).df()
def save(name): plt.savefig(FIG / f'01_{name}.png', bbox_inches='tight')
millions = mt.FuncFormatter(lambda x, _: f'{x/1e6:,.0f}M')
""")

md("### Convert sales CSVs to Parquet (one-time)\nParquet is a compressed column format: the same data is several times smaller and queries run much faster. This is a first piece of the ETL pipeline.")
code("""
for name, folder in [('sales_main', MAIN), ('sales_alt', ALT)]:
    p = CACHE / f'{name}.parquet'
    if not p.exists():
        con.sql(f\"\"\"COPY (SELECT * FROM read_csv_auto('{(folder/'sales_transactions.csv').as_posix()}'))
                    TO '{p.as_posix()}' (FORMAT parquet)\"\"\")
    print(f'{name}: {p.stat().st_size/1e6:,.0f} MB parquet  (csv: {(folder/"sales_transactions.csv").stat().st_size/1e6:,.0f} MB)')

def register(folder, sales_file, suffix=''):
    con.sql(f"CREATE OR REPLACE VIEW sales{suffix} AS SELECT * FROM read_parquet('{(CACHE/sales_file).as_posix()}')")
    for view, file in [('stores','store_master'), ('skus','sku_master'), ('customers','customer_master'),
                       ('promos','promotions'), ('inventory','inventory_snapshot'), ('flags','sku_inventory_flags')]:
        con.sql(f"CREATE OR REPLACE VIEW {view}{suffix} AS SELECT * FROM read_csv_auto('{(folder/(file+'.csv')).as_posix()}')")

register(MAIN, 'sales_main.parquet')
register(ALT, 'sales_alt.parquet', '_alt')
""")

# ---------------------------------------------------------------- 1
md("## 1. Data inventory\nWhat tables exist, how big they are, and whether they contain missing values or duplicate rows.")
code("""
TABLES = ['stores', 'skus', 'customers', 'promos', 'inventory', 'sales', 'flags']
rows = []
for t in TABLES:
    cols = q(f'DESCRIBE {t}').column_name.tolist()
    n = q(f'SELECT COUNT(*) n FROM {t}').n[0]
    nulls = q('SELECT ' + ', '.join(f'COUNT(*) - COUNT("{c}") AS "{c}"' for c in cols) + f' FROM {t}').iloc[0]
    dups = n - q(f'SELECT COUNT(*) n FROM (SELECT DISTINCT * FROM {t})').n[0]
    rows.append({'table': t, 'rows': n, 'columns': len(cols), 'duplicate_rows': dups,
                 'columns_with_nulls': ', '.join(f'{c} ({v:,})' for c, v in nulls.items() if v) or '-'})
overview = pd.DataFrame(rows)
overview
""")
code("""
for t in TABLES:
    display(Markdown(f'**{t}**')); display(q(f'SELECT * FROM {t} LIMIT 3'))
""")
md("""
**How the tables connect (star schema):** `sales` is the fact table; `stores`, `skus`, `customers`, `promos` are dimension tables.
`inventory` is keyed by store × SKU, and `flags` is the answer key for inventory anomalies.

```
stores ─┐
skus ───┼──▶ sales (store_id, sku_id, customer_id, promo_id)
customers ┤
promos ─┘
stores + skus ──▶ inventory        skus ──▶ flags
```
The only nulls in `sales` are in `promo_id`, which is expected: a blank means no promotion was used.
""")

# ---------------------------------------------------------------- 2
md("## 2. Data quality checks\nBefore trusting any chart, check that the data is internally consistent.")
code("""
checks = {
 'Sales rows with unknown store':       "SELECT COUNT(*) FROM sales WHERE store_id NOT IN (SELECT store_id FROM stores)",
 'Sales rows with unknown SKU':         "SELECT COUNT(*) FROM sales WHERE sku_id NOT IN (SELECT sku_id FROM skus)",
 'Sales rows with unknown customer':    "SELECT COUNT(*) FROM sales WHERE customer_id NOT IN (SELECT cust_id FROM customers)",
 'Sales rows with unknown promo':       "SELECT COUNT(*) FROM sales WHERE promo_id IS NOT NULL AND promo_id NOT IN (SELECT promo_id FROM promos)",
 'total_value != qty*price*(1-disc)':   "SELECT COUNT(*) FROM sales WHERE ABS(total_value - quantity*unit_price*(1-discount_pct/100)) > 0.02",
 'Sale price differs from SKU list price': "SELECT COUNT(*) FROM sales s JOIN skus k USING(sku_id) WHERE ABS(s.unit_price - k.unit_price) > 0.01",
 'Quantity <= 0':                       "SELECT COUNT(*) FROM sales WHERE quantity <= 0",
 'Discount without promo_id':           "SELECT COUNT(*) FROM sales WHERE discount_pct > 0 AND promo_id IS NULL",
 'Promo used outside its date window':  "SELECT COUNT(*) FROM sales s JOIN promos p USING(promo_id) WHERE s.date NOT BETWEEN p.start_date AND p.end_date",
 'SKUs with cost >= price':             "SELECT COUNT(*) FROM skus WHERE cost_price >= unit_price",
 'Customers whose 1st purchase is before registration':
     "SELECT COUNT(*) FROM customers c JOIN (SELECT customer_id, MIN(date) d FROM sales GROUP BY 1) f ON c.cust_id = f.customer_id WHERE f.d < c.registration_date",
 'Customers with no purchases':         "SELECT COUNT(*) FROM customers WHERE cust_id NOT IN (SELECT DISTINCT customer_id FROM sales)",
 'Stores opened after 2022-01-01':      "SELECT COUNT(*) FROM stores WHERE opening_date > DATE '2022-01-01'",
 'Exact duplicate sales rows (extra copies)': "SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM sales)) FROM sales",
 'Receipt + SKU pairs that appear more than once':
     "SELECT COUNT(*) FROM (SELECT receipt_id, sku_id FROM sales GROUP BY ALL HAVING COUNT(*) > 1)",
 'Receipts spanning >1 date/store/customer':
     "SELECT COUNT(*) FROM (SELECT receipt_id FROM sales GROUP BY 1 HAVING COUNT(DISTINCT date) > 1 OR COUNT(DISTINCT store_id) > 1 OR COUNT(DISTINCT customer_id) > 1)",
}
qc = pd.DataFrame([(k, int(con.sql(v).fetchone()[0])) for k, v in checks.items()], columns=['check', 'count'])
qc['status'] = np.where(qc['count'] == 0, 'OK', 'Review')
qc
""")
md("""
**Reading the results**
- Keys, prices and the `total_value` formula are consistent, so the data can be joined safely.
- **Duplicate lines:** some receipts contain the same SKU on two separate lines (many are exact copies).
  This could be a double scan or a generator artefact. For modeling we will **aggregate by receipt + SKU** (sum the quantity), which is safe either way.
- **Registration date is unreliable:** many customers bought something before they "registered". Customer tenure should be measured from the **first purchase date**, not from `registration_date`.
- Section 10 shows that *both* folders contain injected inventory anomalies, so the "clean" folder is not truly clean.
""")

# ---------------------------------------------------------------- 3
md("## 3. Business overview (KPIs)")
code("""
kpi = q('''
SELECT COUNT(*)                         AS line_items,
       COUNT(DISTINCT receipt_id)       AS baskets,
       SUM(quantity)                    AS units_sold,
       SUM(total_value)                 AS revenue,
       SUM(total_value)/COUNT(DISTINCT receipt_id) AS avg_basket_value,
       COUNT(*)*1.0/COUNT(DISTINCT receipt_id)     AS items_per_basket,
       AVG(quantity)                    AS avg_qty_per_line,
       MIN(date) AS first_date, MAX(date) AS last_date
FROM sales''')
gp = q('SELECT SUM(s.total_value - s.quantity*k.cost_price) gp FROM sales s JOIN skus k USING(sku_id)').gp[0]
kpi['gross_profit'] = gp; kpi['gross_margin_pct'] = 100 * gp / kpi.revenue
kpi.T.rename(columns={0: 'value'})
""")

# ---------------------------------------------------------------- 4
md("## 4. Sales over time — trend & seasonality\nThis drives the **demand forecasting** model.")
code("""
monthly = q('''SELECT DATE_TRUNC('month', date) AS month_start, SUM(total_value) revenue, SUM(quantity) units,
                      COUNT(DISTINCT receipt_id) baskets FROM sales GROUP BY 1 ORDER BY 1''')
fig, ax = plt.subplots()
ax.plot(monthly.month_start, monthly.revenue, marker='o', ms=3)
ax.yaxis.set_major_formatter(millions); ax.set_title('Monthly revenue (PKR)'); ax.set_xlabel('')
save('04a_monthly_revenue'); plt.show()

yearly = q('''SELECT YEAR(date) yr, SUM(total_value) revenue, SUM(quantity) units, COUNT(DISTINCT receipt_id) baskets
              FROM sales GROUP BY 1 ORDER BY 1''')
for c in ['revenue', 'units', 'baskets']:
    yearly[c + '_growth_pct'] = yearly[c].pct_change() * 100
yearly
""")
code("""
daily = q('''SELECT date, SUM(total_value) revenue, SUM(quantity) units,
                    AVG(CASE WHEN promo_id IS NOT NULL THEN 1 ELSE 0 END) promo_share
             FROM sales GROUP BY 1 ORDER BY 1''')
daily['storewide_promo'] = daily.promo_share > 0.9          # days where (almost) every sale was discounted
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
for ax, col, title in [(axes[0], 'revenue', 'Daily revenue (PKR)'), (axes[1], 'units', 'Daily units sold (demand)')]:
    ax.plot(daily.date, daily[col], alpha=.35, lw=.7)
    ax.plot(daily.date, daily[col].rolling(7).mean(), lw=1.5, label='7-day average')
    ax.fill_between(daily.date, 0, 1, where=daily.storewide_promo, transform=ax.get_xaxis_transform(),
                    color='red', alpha=.12, label='store-wide promotion')
    ax.set_title(title); ax.legend(loc='upper left')
axes[0].yaxis.set_major_formatter(millions)
plt.tight_layout(); save('04b_daily_revenue_vs_units'); plt.show()

eff = daily.groupby('storewide_promo')[['revenue', 'units']].mean()
eff.index = ['normal day', 'store-wide promo day']
display(eff)
print(f"On store-wide promo days revenue changes {eff.revenue.iloc[1]/eff.revenue.iloc[0]-1:+.0%}, "
      f"units change {eff.units.iloc[1]/eff.units.iloc[0]-1:+.0%}  ({daily.storewide_promo.sum()} such days)")
""")
md("""
**Key insight:** the sharp revenue dips are **not** lost demand. They are store-wide promotions where every sale is discounted
(often 30–50% off) but customers buy about the same number of units. Revenue falls ~33% on those days while units barely move. So:
- The forecasting target should be **units (demand)**, not revenue, and the promotion calendar must be an input feature.
- These deep store-wide promotions mostly **give away margin** without growing volume (see section 8).
""")
code("""
seas = q('''SELECT YEAR(date) yr, MONTH(date) m, SUM(quantity) units FROM sales GROUP BY 1,2''')
seas['index'] = seas.units / seas.groupby('yr').units.transform('mean') * 100   # 100 = average month of that year
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.lineplot(data=seas, x='m', y='index', hue='yr', palette='viridis', marker='o', ax=axes[0])
axes[0].axhline(100, color='grey', ls='--'); axes[0].set(title='Seasonality of units sold (100 = average month)', xlabel='Month', xticks=range(1, 13))

dow = q('''SELECT DAYOFWEEK(date) d, DAYNAME(date) AS day_name, SUM(quantity)/COUNT(DISTINCT date) units_per_day
           FROM sales GROUP BY 1,2 ORDER BY 1''')
sns.barplot(data=dow, x='day_name', y='units_per_day', ax=axes[1], color='C0')
axes[1].set(title='Average units per weekday', xlabel='', ylabel='')
plt.tight_layout(); save('04c_seasonality'); plt.show()
display(seas.pivot(index='yr', columns='m', values='index').round(0))
dow.assign(index=dow.units_per_day / dow.units_per_day.mean() * 100)[['day_name', 'index']].T
""")
md("""
- **Yearly seasonality is strong and very stable:** demand is lowest in February (~20% below an average month) and peaks in November (+20%) and December (+35%). The pattern repeats almost identically every year.
- **Revenue growth ≠ demand growth:** units grow a steady 4–6% a year. Revenue jumped 12% in 2025 mainly because there was only one store-wide discount period that year.
- **No weekday effect:** every weekday is within ±1% of average, so day-of-week features will add little.
- **Demand moves in monthly steps:** within a month daily units are fairly flat, then the level jumps at the month boundary (visible in the units chart above). Month-of-year is therefore the single most important calendar feature; trend + yearly seasonality suit Prophet and LSTM.
""")

# ---------------------------------------------------------------- 5
md("## 5. Stores — where is revenue made?")
code("""
st = q('''SELECT st.store_id, st.store_name, st.city, st.store_type, SUM(s.total_value) revenue,
                 COUNT(DISTINCT s.receipt_id) baskets, SUM(s.total_value)/COUNT(DISTINCT s.receipt_id) basket_value
          FROM sales s JOIN stores st USING(store_id) GROUP BY ALL ORDER BY revenue DESC''')
fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
sns.barplot(data=st, x='store_id', y='revenue', hue='store_type', dodge=False, ax=axes[0])
axes[0].yaxis.set_major_formatter(millions); axes[0].tick_params(axis='x', rotation=90)
axes[0].set(title='Revenue by store', xlabel='')
by_type = st.groupby('store_type').agg(stores=('store_id', 'count'), revenue_per_store=('revenue', 'mean'),
                                       basket_value=('basket_value', 'mean')).sort_values('revenue_per_store')
by_type.revenue_per_store.plot.barh(ax=axes[1], color='C0'); axes[1].xaxis.set_major_formatter(millions)
axes[1].set(title='Avg revenue per store, by type', ylabel='')
plt.tight_layout(); save('05_stores'); plt.show()
display(by_type)
st.groupby('city').agg(stores=('store_id', 'count'), revenue=('revenue', 'sum')).sort_values('revenue', ascending=False)
""")

# ---------------------------------------------------------------- 6
md("## 6. Products — categories, margins and bestsellers")
code("""
cat = q('''SELECT k.category, SUM(s.total_value) revenue, SUM(s.quantity) units,
                  SUM(s.total_value - s.quantity*k.cost_price) gross_profit, COUNT(DISTINCT k.sku_id) skus
           FROM sales s JOIN skus k USING(sku_id) GROUP BY 1 ORDER BY revenue DESC''')
cat['margin_pct'] = cat.gross_profit / cat.revenue * 100
cat['revenue_share_pct'] = cat.revenue / cat.revenue.sum() * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=cat, y='category', x='revenue', ax=axes[0], color='C0')
axes[0].xaxis.set_major_formatter(millions); axes[0].set(title='Revenue by category', ylabel='')
sns.barplot(data=cat.sort_values('margin_pct', ascending=False), y='category', x='margin_pct', ax=axes[1], color='C2')
axes[1].set(title='Gross margin % by category', ylabel='', xlabel='%')
plt.tight_layout(); save('06a_categories'); plt.show()
cat
""")
code("""
sku = q('''SELECT k.sku_id, k.sku_name, k.category, k.unit_price, SUM(s.quantity) units, SUM(s.total_value) revenue
           FROM skus k LEFT JOIN sales s USING(sku_id) GROUP BY ALL ORDER BY revenue DESC''')
sku['cum_share'] = sku.revenue.cumsum() / sku.revenue.sum()
sku['sku_rank_pct'] = np.arange(1, len(sku) + 1) / len(sku)
top20 = sku.loc[sku.sku_rank_pct <= .2, 'revenue'].sum() / sku.revenue.sum()
n80 = (sku.cum_share < .8).sum() + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].plot(sku.sku_rank_pct * 100, sku.cum_share * 100)
axes[0].axhline(80, ls='--', c='grey'); axes[0].axvline(n80 / len(sku) * 100, ls='--', c='grey')
axes[0].set(title='Pareto curve: cumulative revenue share', xlabel='% of SKUs (ranked by revenue)', ylabel='% of revenue')
sns.histplot(sku.units, bins=60, ax=axes[1]); axes[1].set(title='Units sold per SKU (4 years)', xlabel='units')
plt.tight_layout(); save('06b_pareto'); plt.show()
print(f'Top 20% of SKUs = {top20:.1%} of revenue;  {n80:,} SKUs ({n80/len(sku):.1%}) make 80% of revenue')
print(f'Units per SKU: min {sku.units.min():,.0f}, median {sku.units.median():,.0f}, max {sku.units.max():,.0f}')
display(sku.head(10)[['sku_id', 'sku_name', 'category', 'unit_price', 'units', 'revenue']])
""")
md("""
**Extreme bestseller skew:** one SKU (`SKU04321`) alone sells over a million units, about **10% of all revenue**,
while the median SKU sells ~2,000 units in 4 years. Most SKUs have low, noisy daily sales.
For forecasting this means: model the top SKUs individually, and forecast the long tail at **category / store level**.
""")
code("""
# Price band vs volume: are cheap products sold in bigger quantities?
sku['price_band'] = pd.qcut(sku.unit_price, 5, labels=['very low', 'low', 'mid', 'high', 'very high'])
sku.groupby('price_band', observed=True).agg(skus=('sku_id', 'count'), avg_price=('unit_price', 'mean'),
                                            avg_units=('units', 'mean'), avg_revenue=('revenue', 'mean'))
""")

# ---------------------------------------------------------------- 7
md("## 7. Customers — who buys, how often, and through which channel\nThis drives **segmentation** and **churn prediction**.")
code("""
cust = q("SELECT * FROM customers")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
sns.histplot(cust.age, bins=30, ax=axes[0]); axes[0].set_title('Age')
for ax, col in zip(axes[1:], ['gender', 'loyalty_segment', 'preferred_channel']):
    order = ['Bronze', 'Silver', 'Gold', 'Platinum'] if col == 'loyalty_segment' else None
    sns.countplot(data=cust, x=col, ax=ax, order=order, color='C0'); ax.set(title=col, xlabel=''); ax.tick_params(axis='x', rotation=20)
plt.tight_layout(); save('07a_customer_profile'); plt.show()
""")
code("""
# Per-customer behaviour = the base table for RFM, segmentation and churn
END = q("SELECT MAX(date) d FROM sales").d[0]
rfm = q(f'''
SELECT c.cust_id, c.age, c.gender, c.city, c.loyalty_segment, c.preferred_channel,
       DATE_DIFF('day', MAX(s.date), DATE '{END:%Y-%m-%d}') AS recency_days,
       COUNT(DISTINCT s.receipt_id)                       AS frequency,
       SUM(s.total_value)                                 AS monetary,
       SUM(s.total_value)/COUNT(DISTINCT s.receipt_id)    AS avg_basket_value,
       DATE_DIFF('day', MIN(s.date), MAX(s.date))         AS active_span_days,
       AVG(CASE WHEN s.channel = c.preferred_channel THEN 1 ELSE 0 END) AS pref_channel_share,
       AVG(CASE WHEN s.promo_id IS NOT NULL THEN 1 ELSE 0 END)          AS promo_share,
       COUNT(DISTINCT s.store_id)                         AS stores_used
FROM customers c JOIN sales s ON s.customer_id = c.cust_id GROUP BY ALL''')
rfm.describe().T
""")
code("""
order = ['Bronze', 'Silver', 'Gold', 'Platinum']
seg = rfm.groupby('loyalty_segment').agg(customers=('cust_id', 'count'), avg_baskets=('frequency', 'mean'),
        avg_spend=('monetary', 'mean'), avg_basket_value=('avg_basket_value', 'mean'),
        avg_recency_days=('recency_days', 'mean')).reindex(order)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
seg.avg_baskets.plot.bar(ax=axes[0], color='C0', title='Avg baskets per customer (4 yrs)')
seg.avg_spend.plot.bar(ax=axes[1], color='C1', title='Avg spend per customer'); axes[1].yaxis.set_major_formatter(millions)
sns.histplot(rfm.frequency, bins=60, ax=axes[2]); axes[2].set_title('Baskets per customer (all)')
for a in axes[:2]: a.tick_params(axis='x', rotation=0); a.set_xlabel('')
plt.tight_layout(); save('07b_loyalty'); plt.show()
display(seg)
print(f"Customers shop on their preferred channel {rfm.pref_channel_share.mean():.1%} of the time")
print(f"Top 20% of customers by spend = {rfm.monetary.nlargest(len(rfm)//5).sum()/rfm.monetary.sum():.1%} of revenue")
""")
md("### Is there churn in this data?\nChurn usually means a customer *stops buying*. Let's check recency and the longest gap between purchases.")
code("""
gaps = q('''WITH d AS (SELECT DISTINCT customer_id, date FROM sales),
                 g AS (SELECT customer_id,
                              DATE_DIFF('day', LAG(date) OVER (PARTITION BY customer_id ORDER BY date), date) gap FROM d)
            SELECT customer_id, MAX(gap) max_gap_days, AVG(gap) avg_gap_days FROM g GROUP BY 1''')
# Spend change: Q4-2025 vs Q4-2024 per customer (same season, so seasonality cancels out)
yoy = q('''SELECT customer_id,
                  SUM(total_value) FILTER (WHERE date BETWEEN '2024-10-01' AND '2024-12-31') q4_2024,
                  SUM(total_value) FILTER (WHERE date BETWEEN '2025-10-01' AND '2025-12-31') q4_2025
           FROM sales GROUP BY 1''')
yoy['change_pct'] = (yoy.q4_2025 / yoy.q4_2024 - 1) * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(rfm.recency_days, bins=30, ax=axes[0]); axes[0].set(title='Days since last purchase (31-Dec-2025)', xlabel='days')
sns.histplot(gaps.max_gap_days, bins=40, ax=axes[1]); axes[1].set(title='Longest gap between purchases', xlabel='days')
sns.histplot(yoy.change_pct, bins=60, ax=axes[2]); axes[2].axvline(-30, c='red', ls='--')
axes[2].set(title='Spend change Q4-2025 vs Q4-2024', xlabel='%')
plt.tight_layout(); save('07c_churn_check'); plt.show()

print(f"Max days since last purchase: {rfm.recency_days.max()}  |  average gap between shopping days: {gaps.avg_gap_days.mean():.1f} days")
print(f"Longest gap any customer ever had between purchases: {gaps.max_gap_days.max()} days")
for t in [-20, -30, -40]:
    print(f"Customers whose Q4 spend fell by more than {-t}% year-on-year: {(yoy.change_pct < t).mean():.1%}")
""")
code("""
# Customer activity over time: are customers actually leaving?
act = q('''SELECT DATE_TRUNC('quarter', date) AS qtr, COUNT(DISTINCT customer_id) active_customers,
                  COUNT(DISTINCT receipt_id)*1.0/COUNT(DISTINCT customer_id) baskets_per_active
           FROM sales GROUP BY 1 ORDER BY 1''')
first = q("SELECT DATE_TRUNC('quarter', MIN(date)) AS qtr, COUNT(*) new_customers FROM sales GROUP BY customer_id")
act = act.merge(first.groupby('qtr').size().rename('new_customers').reset_index(), on='qtr', how='left').fillna(0)
ch = q("SELECT channel, COUNT(*) lines, SUM(total_value) revenue, SUM(total_value)/COUNT(DISTINCT receipt_id) basket_value FROM sales GROUP BY 1 ORDER BY revenue DESC")
ch['revenue_share_pct'] = ch.revenue / ch.revenue.sum() * 100
display(act); display(ch)
""")
code("""
# Sample check: quarter-to-quarter retention (share of customers active in Q who are also active in Q+1)
ret = q('''WITH a AS (SELECT DISTINCT customer_id, DATE_TRUNC('quarter', date) AS qtr FROM sales)
           SELECT a.qtr, COUNT(*) active, COUNT(b.customer_id) retained_next_q
           FROM a LEFT JOIN a b ON a.customer_id = b.customer_id AND b.qtr = a.qtr + INTERVAL 3 MONTH
           GROUP BY 1 ORDER BY 1''')
ret['retention_pct'] = ret.retained_next_q / ret.active * 100
ret.iloc[:-1]
""")
md("""
**Important finding for the churn model:** nobody leaves. All 10,000 customers buy in every single quarter
(100% retention), all customers shop at all 30 stores, and no one goes more than about a month without buying.
A classic "stopped buying" churn label would have **zero positive cases**.

The churn target therefore has to be **defined by us**. The best option for this data is
**"spend-decline churn"**: a customer whose spend in the next quarter drops by more than X% (e.g. 30%) versus the
same quarter last year. That is a realistic early-warning signal ("customer is drifting away") and the histogram above shows
there is enough variation to learn from. We will confirm the exact threshold in the churn step.
""")

# ---------------------------------------------------------------- 8
md("## 8. Promotions — do discounts change behaviour?")
code("""
pr = q('''SELECT p.promo_type, p.target_type, COUNT(DISTINCT p.promo_id) promos, COUNT(s.sku_id) lines,
                 AVG(s.discount_pct) avg_disc, SUM(s.quantity*s.unit_price - s.total_value) discount_given
          FROM promos p LEFT JOIN sales s USING(promo_id) GROUP BY ALL ORDER BY lines DESC''')
display(pr)
comp = q('''SELECT promo_id IS NOT NULL AS on_promo, COUNT(*) lines, AVG(quantity) avg_qty,
                   AVG(total_value) avg_line_value, SUM(total_value) revenue FROM sales GROUP BY 1''')
comp['share_of_lines_pct'] = comp.lines / comp.lines.sum() * 100
comp
""")
code("""
# Promo lift for store-wide promotions: units/day during the promo vs. the 14 days before it
units_by_day = daily.set_index('date').units
rows = []
for p in q("SELECT * FROM promos WHERE target_type = 'All' ORDER BY start_date").itertuples():
    s, e = pd.Timestamp(p.start_date), pd.Timestamp(p.end_date)
    during = units_by_day[s:e].mean()
    before = units_by_day[s - pd.Timedelta(days=14): s - pd.Timedelta(days=1)].mean()
    rows.append({'promo_id': p.promo_id, 'promo_name': p.promo_name, 'start': s.date(), 'days': (e - s).days + 1,
                 'promo_type': p.promo_type, 'discount_pct': p.discount_pct, 'unit_lift_pct': (during / before - 1) * 100})
lift = pd.DataFrame(rows)
display(lift)
print(f"Average unit lift of store-wide promos: {lift.unit_lift_pct.mean():+.1f}%  "
      f"(correlation between discount depth and lift: {lift.discount_pct.corr(lift.unit_lift_pct):+.2f})")
""")
md("""
**Promotions barely move demand.** Promoted lines have the same average quantity as normal lines, and store-wide
promotions lift units only slightly, whatever the discount depth. In this data, discounts mainly reduce revenue and margin.
This is a useful business message ("re-think deep store-wide discounts") and means promo features will matter more for
**revenue** forecasts than for **unit** forecasts.
""")
code("""
pm = q('''SELECT DATE_TRUNC('month', date) AS month_start, AVG(CASE WHEN promo_id IS NOT NULL THEN 1 ELSE 0 END)*100 promo_share
          FROM sales GROUP BY 1 ORDER BY 1''')
fig, ax = plt.subplots(); ax.bar(pm.month_start, pm.promo_share, width=25)
ax.set(title='% of sales lines with a promotion, by month', ylabel='%'); save('08_promo_share'); plt.show()
""")

# ---------------------------------------------------------------- 9
md("## 9. Baskets — what is bought together?\nUseful for cross-selling and market-basket analysis.")
code("""
bs = q("SELECT n, COUNT(*) baskets FROM (SELECT receipt_id, COUNT(*) n FROM sales GROUP BY 1) GROUP BY 1 ORDER BY 1")
bs['share_pct'] = bs.baskets / bs.baskets.sum() * 100
display(bs)
pairs = q('''
WITH x AS (SELECT s.receipt_id, k.category FROM sales s JOIN skus k USING(sku_id))
SELECT a.category cat_a, b.category cat_b, COUNT(*) together
FROM x a JOIN x b ON a.receipt_id = b.receipt_id AND a.category < b.category
GROUP BY 1,2 ORDER BY 3 DESC''')
mat = pairs.pivot(index='cat_a', columns='cat_b', values='together')
plt.figure(figsize=(11, 8)); sns.heatmap(mat, cmap='Blues', fmt='.0f')
plt.title('How often two categories appear in the same basket'); save('09_basket_pairs'); plt.show()
pairs.head(10)
""")

# ---------------------------------------------------------------- 10
md("## 10. Inventory — stock health and the anomaly answer key\nThis drives **inventory optimization** (stockouts and dead stock).")
code("""
inv = q('''SELECT *, CASE WHEN stock_on_hand = 0 THEN '1 Out of stock'
                          WHEN stock_on_hand < safety_stock THEN '2 Below safety stock'
                          WHEN stock_on_hand < reorder_point THEN '3 Reorder now'
                          WHEN stock_on_hand > 3*reorder_point THEN '5 Overstock (>3x reorder pt)'
                          ELSE '4 Healthy' END status
           FROM inventory''')
print(f"Inventory rows: {len(inv):,}  covering {inv.sku_id.nunique():,} SKUs and {inv.store_id.nunique()} stores "
      f"(of {30*5000:,} possible store x SKU pairs)")
status = inv.status.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 3.5)); status.plot.barh(ax=ax, color='C0'); ax.invert_yaxis()
ax.set(title='Stock status of store x SKU rows', xlabel='rows'); save('10a_stock_status'); plt.show()
(status / len(inv) * 100).round(1).rename('% of rows')
""")
code("""
fl = q('''SELECT flag, COUNT(*) skus, AVG(LEN(STRING_SPLIT(affected_stores, ';'))) avg_stores_affected,
                 MIN(window_start) earliest_start, MAX(window_end) latest_end,
                 AVG(DATE_DIFF('day', window_start, window_end)) avg_window_days
          FROM flags GROUP BY 1''')
display(fl)
# Where do flagged SKUs sit in the sales ranking?
rank = sku[['sku_id', 'units']].assign(rank=np.arange(1, len(sku) + 1))
rank = rank.sort_values('units', ascending=False).assign(volume_rank=np.arange(1, len(sku) + 1))
flagged = q("SELECT sku_id, flag FROM flags").merge(rank, on='sku_id')
flagged.groupby('flag').volume_rank.describe()
""")
md("""
- The inventory snapshot only covers ~21k of 150k possible store × SKU pairs, and it is unhealthy: a large share is
  **overstocked** (>3× reorder point) while a noticeable share is **out of stock**.
- `STOCKOUT_RISK` SKUs are exactly the **top 200 sellers**; `SLOW_MOVER` SKUs come from the **bottom ~1,000**.
  Stockout windows are all in the last ~75 days of 2025.
""")
code("""
# Show one injected stockout: daily units of a flagged SKU in affected vs. unaffected stores
ex = q("SELECT sku_id, affected_stores, window_start, window_end FROM flags WHERE flag='STOCKOUT_RISK' LIMIT 1").iloc[0]
aff = ex.affected_stores.split(';')
w = q(f'''SELECT date, CASE WHEN store_id IN ({",".join(f"'{s}'" for s in aff)}) THEN 'Affected stores' ELSE 'Other stores' END grp,
                 SUM(quantity) units
          FROM sales WHERE sku_id = '{ex.sku_id}' AND date >= DATE '2025-09-01' GROUP BY ALL ORDER BY 1''')
w['units_per_store'] = w.units / np.where(w.grp == 'Affected stores', len(aff), 30 - len(aff))
w = w.pivot(index='date', columns='grp', values='units_per_store').fillna(0)   # missing day = zero sales
fig, ax = plt.subplots()
w.rolling(3).mean().plot(ax=ax)
ax.axvspan(ex.window_start, ex.window_end, color='red', alpha=.12, label='stockout window')
ax.set(title=f'{ex.sku_id}: daily units per store (3-day avg) — affected stores drop to zero', xlabel='', ylabel='units / store')
ax.legend(); save('10b_stockout_example'); plt.show()
""")
md("### Comparing the two dataset folders")
code("""
def window_sales(sfx):
    return con.sql(f'''WITH f AS (SELECT sku_id, TRIM(UNNEST(STRING_SPLIT(affected_stores, ';'))) store_id, window_start, window_end
                                  FROM flags{sfx} WHERE flag = 'STOCKOUT_RISK')
                       SELECT COUNT(*) FROM sales{sfx} s JOIN f USING(sku_id, store_id)
                       WHERE s.date BETWEEN f.window_start AND f.window_end''').fetchone()[0]
cmp = pd.DataFrame({
  'sales rows':        [q('SELECT COUNT(*) n FROM sales').n[0], q('SELECT COUNT(*) n FROM sales_alt').n[0]],
  'revenue':           [q('SELECT SUM(total_value) v FROM sales').v[0], q('SELECT SUM(total_value) v FROM sales_alt').v[0]],
  'inventory rows':    [len(inv), q('SELECT COUNT(*) n FROM inventory_alt').n[0]],
  'flagged SKUs':      [q('SELECT COUNT(*) n FROM flags').n[0], q('SELECT COUNT(*) n FROM flags_alt').n[0]],
  'sales inside own stockout windows': [window_sales(''), window_sales('_alt')],
}, index=['retail_clean_dataset', 'retail_contaminated_dataset']).T
display(cmp)
same = q('''SELECT a.flag, COUNT(*) skus_flagged_in_both FROM flags a JOIN flags_alt b USING(sku_id, flag) GROUP BY 1 ORDER BY 1''')
same
""")
md("""
**Interpretation:** both folders have ~0.3% fewer than 10M rows and **zero** sales inside their own flagged stockout windows.
So **both** folders contain injected anomalies (with different stores/windows), and neither is a true "no-anomaly" baseline.
For modeling we can use either folder; each one's `sku_inventory_flags.csv` is its answer key. The last ~75 days of 2025
contain the stockout gaps, which matters when we split train/test for forecasting.
""")

# ---------------------------------------------------------------- 11
md("""
## 11. Key findings

| # | Finding | Why it matters |
|---|---|---|
| 1 | Data is consistent: no broken keys, prices and `total_value` all check out | Tables can be joined safely |
| 2 | ~13k exact duplicate sales lines; `registration_date` is after the first purchase for 37% of customers | Aggregate by receipt + SKU; measure tenure from first purchase |
| 3 | **Both** folders contain injected stockouts and slow movers (neither is truly "clean") | Each folder's flags file is its own answer key |
| 4 | Revenue ≈ PKR 10.9 bn over 4 years, 27.6% gross margin, ~1.9 items and ~PKR 2,110 per basket | Headline KPIs for the dashboard |
| 5 | Demand grows a steady 4–6% a year with strong, stable yearly seasonality (Feb low, Nov–Dec peak) and no weekday effect | Good fit for Prophet and LSTM; forecast **units** |
| 6 | Store-wide promotions cut revenue ~33% but barely change units | Promo calendar is a must-have forecasting input; deep discounts give away margin |
| 7 | Extreme product skew: 1 SKU ≈ 10% of revenue; 27% of SKUs make 80% of revenue | Forecast top SKUs individually, long tail at category level |
| 8 | Hypermarkets earn ~2× a convenience store; basket value is identical across store types, channels and loyalty tiers | Differences come from **visit frequency**, not basket size |
| 9 | Loyalty tier strongly drives frequency (Platinum ≈ 3.3× Bronze baskets) | Strong segmentation signal |
| 10 | **No natural churn**: every customer buys every quarter | Churn label must be defined as a spend decline |
| 11 | Inventory snapshot: large overstock share and some out-of-stock rows; flags = top-200 sellers (stockout) and bottom sellers (slow movers) | Clear, measurable target for the inventory module |

## 12. Feature catalogue for the next steps

**Demand forecasting** (target: daily/weekly units per SKU, category or store)
- Calendar: day of year, week of year, month, quarter, days to month end, year (trend)
- Seasonality: month index from history; Nov–Dec peak flag; February trough flag
- Promotions: promo active flag, discount %, promo type, store-wide flag, days since promo start
- Lags & rolling stats: units 7/14/28/364 days ago, 7/28-day rolling mean and std
- Product: category, subcategory, brand, price band, bestseller rank
- Store: store type, city
- Data hygiene: mark stockout windows (zero sales ≠ zero demand) so they don't teach the model wrong patterns

**Customer segmentation**
- RFM: recency, frequency (baskets), monetary (spend)
- Average basket value, items per basket, average gap between visits
- Category mix (% spend per category), promo share, preferred-channel share
- Demographics: age, gender, city, loyalty tier

**Churn (spend-decline) prediction** (target: next-quarter spend falls >30% vs same quarter last year)
- Spend and visits in the last 30/90/180 days, and their trends (last 90 vs previous 90 days)
- Change in basket value and visit gap over time
- Promo dependence, channel mix shift, category diversity
- Tenure from first purchase, loyalty tier, age, city

**Inventory optimization**
- Stock on hand vs reorder point and safety stock (ratios)
- Days of cover = stock on hand ÷ average daily units for that store × SKU
- Days since last restock; recent sales velocity vs long-run average
- Sudden drop to zero sales for a normally fast seller (stockout signal)
- Evaluation: precision / recall against `sku_inventory_flags.csv`
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = Path(__file__).resolve().parent.parent / 'notebooks' / '01_eda.ipynb'
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
