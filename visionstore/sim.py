"""VisionStore AI - Retail simulation & BI model (seeded, deterministic)."""
import numpy as np
import pandas as pd

def hourly_traffic(seed=42, visitors=1850, hours=(9, 22)):
    rng = np.random.default_rng(seed)
    hrs = list(range(*hours))
    shape = np.array([1,1.4,1.8,2.1,2.0,1.9,2.4,3.1,3.8,4.4,3.9,2.6,1.6][:len(hrs)])
    w = shape / shape.sum()
    counts = rng.multinomial(visitors, w)
    return pd.DataFrame({"hour": hrs, "visitors": counts})

def demographics(seed=42, visitors=1850):
    rng = np.random.default_rng(seed)
    genders = rng.choice(["Female", "Male"], visitors, p=[0.58, 0.42])
    ages = rng.choice(["0-17", "18-25", "26-35", "36-50", "51+"], visitors, p=[0.05, 0.24, 0.34, 0.27, 0.10])
    dwell = np.clip(rng.gamma(2.4, 4.2, visitors), 0.5, 75)
    returning = rng.random(visitors) < 0.31
    df = pd.DataFrame({"gender": genders, "age": ages, "dwell_min": dwell.round(1), "returning": returning})
    return df

def aisle_compare(seed=42):
    rng = np.random.default_rng(seed)
    aisles = ["Entrance", "Aisle A", "Aisle B", "Aisle C", "Fitting Rooms", "Checkout"]
    before = rng.integers(120, 900, len(aisles))
    after = np.clip(before * rng.uniform(0.85, 1.35, len(aisles)), 0, None).astype(int)
    dwell = rng.uniform(0.6, 6.0, len(aisles)).round(1)
    return pd.DataFrame({"zone": aisles, "traffic_before": before, "traffic_after": after,
                         "dwell_min": dwell, "lift_pct": ((after - before) / np.maximum(before, 1) * 100).round(1)})

def roi_model(traffic=1850, conversion=0.22, basket=650.0, lift=0.18, currency="TL"):
    base_rev = traffic * conversion * basket
    new_conv = min(conversion * (1 + lift), 0.95)
    new_rev = traffic * new_conv * basket
    return {"base_revenue": base_rev, "projected_revenue": new_rev,
            "uplift": new_rev - base_rev, "new_conversion": new_conv}

def staff_optimizer(tr_df, staff_per_150=1):
    tr_df = tr_df.copy()
    tr_df["staff_needed"] = np.ceil(tr_df["visitors"] / 150).astype(int) * staff_per_150
    return tr_df

ALERT_RULES = {
    "loitering": "Dwell > 8 min in non-shopping zone",
    "crowd": "Density > 12 persons in one zone",
    "queue": "Queue wait estimate with > 3 persons",
    "intrusion": "Motion detected off-hours (after close)",
}
