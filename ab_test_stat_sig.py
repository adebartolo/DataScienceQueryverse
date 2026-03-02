# A/B Testing for determining Lift and Stats Sig
# Handles multiple metrics, multi-level grouping, and optional filters.
# Differentiates between proportion metrics (CTR → z-test) and ratio/continuous metrics (CPC, CPM → t-test).
# Outputs a clean, ordered table with lifts, p-values, and flags for warnings or significance

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------
# Simulate data
# -------------------------------
np.random.seed(42)
days = pd.date_range("2026-02-01", periods=14)
slots, types, sections = ["banner","multi1","multi2","multi3"], ["article","homepage"], ["news","sports","finance"]

GAM_MW = pd.DataFrame([
    {
        "date": d,
        "app_name": v,
        "ad_slot": s,
        "page_type": pt,
        "page_section": ps,
        "clicks": np.random.poisson(50 if v=="mw-c" else 55),
        "impressions": np.random.poisson(1000),
        "revenue": np.random.normal(120 if v=="mw-c" else 130, 10)
    }
    for d in days for v in ["mw-c","mw-t"] for s in slots for pt in types for ps in sections
])
GAM_MW["capacity"] = GAM_MW["impressions"] + 5

DIANOMI_MW = GAM_MW.copy()
DIANOMI_DT = GAM_MW.copy()

# -------------------------------
# Helpers
# -------------------------------
def run_ttest(c, v):
    c = c.replace([np.inf, -np.inf], np.nan).dropna()
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    if len(c) < 2 or len(v) < 2:
        return np.nan
    return ttest_ind(c, v, equal_var=False)[1]  # Welch t-test


def apply_filters(df, filt):
    for col, val in (filt or {}).items():
        if col not in df:
            continue
        if isinstance(val, tuple):
            df = df[(df[col] >= val[0]) & (df[col] <= val[1])]
        elif isinstance(val, list):
            df = df[df[col].isin(val)]
        else:
            df = df[df[col] == val]
    return df

# -------------------------------
# Source map
# -------------------------------
source_map = {
    "GAM_MW": ("mw-c", "mw-t"),
    "GAM_DT": ("dt-c", "dt-t"),
    "DIANOMI_MW": ("mw-c", "mw-t"),
    "DIANOMI_DT": ("mw-c", "mw-t")
}

# -------------------------------
# Metric calculation
# -------------------------------
def metric_calc(c, v, m):
    m_l = m.lower()

    # ---- Proportion metric ----
    if m_l == "ctr":
        c_daily = c.clicks / c.impressions.replace(0, np.nan)
        v_daily = v.clicks / v.impressions.replace(0, np.nan)

        p_val = proportions_ztest(
            [c.clicks.sum(), v.clicks.sum()],
            [c.impressions.sum(), v.impressions.sum()]
        )[1]

    # ---- Ratio metrics (Welch t-test on daily ratios) ----
    elif m_l in ["cpm", "rpm"]:
        c_daily = (c.revenue / c.impressions.replace(0, np.nan)) * 1000
        v_daily = (v.revenue / v.impressions.replace(0, np.nan)) * 1000
        p_val = run_ttest(c_daily, v_daily)

    elif m_l in ["ecpm"]:
        c_daily = (c.revenue / c.capacity.replace(0, np.nan)) * 1000
        v_daily = (v.revenue / v.capacity.replace(0, np.nan)) * 1000
        p_val = run_ttest(c_daily, v_daily)

    elif m_l == "cpc":
        c_daily = c.revenue / c.clicks.replace(0, np.nan)
        v_daily = v.revenue / v.clicks.replace(0, np.nan)
        p_val = run_ttest(c_daily, v_daily)

    # ---- Mean metrics ----
    elif m_l in ["revenue", "clicks", "impressions", "capacity"]:
        c_daily = c[m]
        v_daily = v[m]
        p_val = run_ttest(c_daily, v_daily)

    else:
        raise ValueError(f"Unknown metric: {m}")

    # Summary stats
    c_val, v_val = c_daily.mean(), v_daily.mean()
    c_std, v_std = c_daily.std(), v_daily.std()

    abs_l = v_val - c_val
    rel_lift = (v_val - c_val) / c_val if c_val else np.nan

    return c_val, v_val, c_std, v_std, abs_l, rel_lift, p_val, False


# -------------------------------
# A/B Test Engine
# -------------------------------
def ab_test_engine(source, metrics, variant_col="app_name",
                   group_col=None, filters=None,
                   alpha=0.05, date_col="date"):

    if source not in globals():
        raise ValueError(f"DataFrame '{source}' does not exist.")
    df = globals()[source].copy()

    df = apply_filters(df, filters)
    df[date_col] = pd.to_datetime(df[date_col])

    if source not in source_map:
        raise ValueError(f"Source '{source}' missing from source_map")
    ctrl, var = source_map[source]

    agg_cols = [date_col, variant_col] + ([c for c in (group_col or []) if c in df])
    df_agg = df.groupby(agg_cols)[
        ["clicks","impressions","capacity","revenue"]
    ].sum().reset_index()

    grouped = [(("overall",), df_agg)] if not group_col else list(df_agg.groupby(group_col))
    res = []

    for gk, temp in grouped:
        gd = {"group":"overall"} if not group_col else dict(
            zip(group_col, gk if isinstance(gk, tuple) else (gk,))
        )

        c = temp[temp[variant_col] == ctrl]
        v = temp[temp[variant_col] == var]
        if c.empty or v.empty:
            continue

        for m in metrics:
            c_val, v_val, c_std, v_std, abs_l, rel_l, p, warn = metric_calc(c, v, m)

            res.append({
                "min_date": temp[date_col].min(),
                "max_date": temp[date_col].max(),
                "source": source,
                "control_name": ctrl,
                "variant_name": var,
                **gd,
                "metric": m,
                "control": c_val,
                "control_std": c_std,
                "variant": v_val,
                "variant_std": v_std,
                "absolute_lift": abs_l,
                "relative_lift": rel_l,
                "p_value": p,
                "significant": bool(p < alpha) if pd.notna(p) else False,
                "warning_flag": warn
            })

    return pd.DataFrame(res)


# -------------------------------
# Example Run
# -------------------------------
res = ab_test_engine(
    source="GAM_MW",
    metrics=[
            "capacity","impressions","clicks","cpm","ecpm","ctr","cpc","revenue",
            ], 
    group_col=["page_section"],
    filters={"page_type":["article"], "page_section":["news","sports"]
             #, "date": ("2026-02-01", "2026-02-07") 
            }
)

print(res.to_string(index=False))
