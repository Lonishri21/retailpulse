"""Inventory policy: safety stock, reorder point, EOQ, ABC class and a weekly simulation."""
import numpy as np
import pandas as pd


def safety_stock(z, sigma, lead_weeks=1):
    return z * sigma * np.sqrt(lead_weeks)


def reorder_point(weekly_demand, ss, lead_weeks=1):
    return weekly_demand * lead_weeks + ss


def eoq(weekly_demand, order_cost, holding_rate, unit_cost):
    """EOQ, with annual demand = 52 x weekly demand."""
    return np.sqrt(2 * 52 * weekly_demand * order_cost / (holding_rate * unit_cost))


def abc_class(revenue: pd.Series) -> pd.Series:
    """A = items making the first 80% of revenue, B = next 15%, C = last 5%."""
    share = revenue.sort_values(ascending=False).cumsum() / revenue.sum()
    prev = share.shift(fill_value=0)          # share before adding the item
    cls = np.select([prev < 0.80, prev < 0.95], ["A", "B"], "C")
    return pd.Series(cls, index=share.index).reindex(revenue.index)


def simulate(demand, rop, q, start, lead_weeks=1):
    """Weekly reorder-point policy: when stock + on order <= rop, order the smallest multiple
    of q that gets back above it. rop and q can be numbers or one value per week."""
    demand = np.asarray(demand, float)
    rop, q = np.broadcast_to(rop, demand.shape), np.broadcast_to(q, demand.shape)
    stock, pipeline = float(start), [0.0] * lead_weeks
    levels, orders, sold_total, short_weeks = [], [], 0.0, 0
    for t, d in enumerate(demand):
        stock += pipeline.pop(0)
        sold = min(stock, d)
        sold_total += sold
        short_weeks += d > stock
        stock -= sold
        position = stock + sum(pipeline)
        order = np.ceil((rop[t] - position) / q[t] + 1e-9) * q[t] if position <= rop[t] else 0.0
        pipeline.append(order)
        levels.append(stock)
        orders.append(order)
    return {"fill_rate_%": 100 * sold_total / max(demand.sum(), 1), "stockout_weeks": int(short_weeks),
            "avg_inventory": float(np.mean(levels)), "orders": int(np.count_nonzero(orders)),
            "levels": np.array(levels), "order_qty": np.array(orders)}


def simulate_naive(demand, start, lead_weeks=1):
    """Each week, reorder whatever sold."""
    demand = np.asarray(demand, float)
    stock, pipeline = float(start), [0.0] * lead_weeks
    levels, orders, sold_total, short_weeks = [], [], 0.0, 0
    for d in demand:
        stock += pipeline.pop(0)
        sold = min(stock, d)
        sold_total += sold
        short_weeks += d > stock
        stock -= sold
        pipeline.append(sold)
        levels.append(stock)
        orders.append(sold)
    return {"fill_rate_%": 100 * sold_total / max(demand.sum(), 1), "stockout_weeks": int(short_weeks),
            "avg_inventory": float(np.mean(levels)), "orders": int(np.count_nonzero(orders)),
            "levels": np.array(levels), "order_qty": np.array(orders)}
