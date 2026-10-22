# Dataset

Synthetic multi-store retail data: 30 stores, 5,000 SKUs, 10,000 customers, sales from 2022 to 2025 (prices in PKR). 100% synthetic.

## Files

| File | `retail_clean_dataset/` | `retail_contaminated_dataset/` | Columns |
|---|---|---|---|
| `store_master.csv` | 30 | 30 | 5 |
| `sku_master.csv` | 5,000 | 5,000 | 7 |
| `customer_master.csv` | 10,000 | 10,000 | 7 |
| `promotions.csv` | 100 | 100 | 8 |
| `inventory_snapshot.csv` | 21,228 | 26,408 | 6 |
| `sales_transactions.csv` | 9,972,038 | 9,945,511 | 11 |
| `sku_inventory_flags.csv` | 600 | 600 | 6 |

- `retail_clean_dataset/` is the main dataset; the other folder is used only for comparison in the EDA.
- Both folders contain injected stockouts (with the matching sales removed) and slow movers. Each folder's `sku_inventory_flags.csv` is its answer key: 200 `STOCKOUT_RISK` and 400 `SLOW_MOVER` rows.
- About 13k sales lines are exact duplicates; Step 2 removes them.
- The two `sales_transactions.csv` files are not in git (too large); see "Data setup" in the [README](../README.md).

## Column Reference

__store_master.csv__
| Column | Description |
|---|---|
| store_id | Unique store identifier (e.g. `ST01`) |
| store_name | Store display name |
| city | City the store operates in |
| store_type | Hypermarket / Supermarket / Convenience Store / Express Store |
| opening_date | Date the store opened |

__sku_master.csv__
| Column | Description |
|---|---|
| sku_id | Unique product identifier (e.g. `SKU00001`) |
| sku_name | Product display name |
| category | Top-level product category |
| subcategory | Product subcategory |
| unit_price | Retail selling price |
| cost_price | Cost to the retailer (always < unit_price) |
| brand | Product brand |

__customer_master.csv__
| Column | Description |
|---|---|
| cust_id | Unique customer identifier (e.g. `CUST00001`) |
| age | Customer age |
| gender | Male / Female / Other |
| city | Customer's city |
| loyalty_segment | Bronze / Silver / Gold / Platinum |
| preferred_channel | In-Store / Online / Mobile App |
| registration_date | Date the customer registered |

__promotions.csv__
| Column | Description |
|---|---|
| promo_id | Unique promotion identifier |
| promo_name | Campaign name |
| start_date / end_date | Active date range |
| discount_pct | Discount percentage applied |
| promo_type | Percentage Discount / Flat Discount / BOGO / Bundle Offer / Clearance |
| target_type | What the promo applies to: Category / Brand / SKU / All |
| target_value | The specific category, brand, sku_id, or "All" being targeted |

__inventory_snapshot.csv__
| Column | Description |
|---|---|
| store_id / sku_id | Which store and product this row describes |
| stock_on_hand | Current units in stock |
| reorder_point | Stock level that should trigger a reorder |
| safety_stock | Minimum buffer stock |
| last_restock_date | Date of the most recent restock |

__sales_transactions.csv__
| Column | Description |
|---|---|
| date | Transaction date |
| receipt_id | Groups line items purchased in the same basket/visit (1–5 items) |
| store_id | Store where the sale occurred |
| sku_id | Product sold |
| customer_id | Customer who made the purchase |
| quantity | Units purchased |
| unit_price | Price per unit at time of sale |
| total_value | `quantity × unit_price × (1 − discount_pct/100)` |
| channel | In-Store / Online / Mobile App |
| discount_pct | Discount applied (0 if none) |
| promo_id | Promotion applied, blank if none |

__sku_inventory_flags.csv__ _(answer key for injected anomalies)_
| Column | Description |
|---|---|
| sku_id | The flagged product |
| flag | `STOCKOUT_RISK` or `SLOW_MOVER` |
| affected_stores | `;`-separated list of `store_id`s where the anomaly was applied |
| window_start / window_end | Stockout date window (blank for `SLOW_MOVER` rows) |
| notes | Plain-language reason the SKU was flagged |

## Relationships

```ini
store_master.store_id        ──┐
sku_master.sku_id             ─┼──▶ sales_transactions.csv
customer_master.cust_id       ─┤     (store_id, sku_id, customer_id, promo_id)
promotions.promo_id           ─┘

store_master.store_id  ──┐
sku_master.sku_id        ─┴──▶ inventory_snapshot.csv

sku_master.sku_id  ──▶ sku_inventory_flags.csv
```

## License

Apache License 2.0.
