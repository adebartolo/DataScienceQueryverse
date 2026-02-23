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
days = pd.date_range(start="2026-02-01", periods=14)
slots = ["banner", "multi1", "multi2", "multi3"]
page_types = ["article", "homepage"]
page_sections = ["news", "sports", "finance"]

rows = []
for day in days:
    for variant in ["mw-c", "mw-t"]:
        for slot in slots:
            for page_type in page_types:
                for section in page_sections:
                    clicks = np.random.poisson(50 if variant == "mw-c" else 55)
                    impressions = np.random.poisson(1000)
                    capacity = impressions + 5
                    revenue = np.random.normal(120 if variant == "mw-c" else 130, 10)

                    rows.append({
                        "date": day,
                        "app_name": variant,
                        "ad_slot": slot,
                        "page_type": page_type,
                        "page_section": section,
                        "clicks": clicks,
                        "impressions": impressions,
                        "capacity": capacity,
                        "revenue": revenue
                    })

DF_GAM = pd.DataFrame(rows)

# -------------------------------
# Helper functions
# -------------------------------
def run_ttest(c_series, v_series):
    c_series = c_series.replace([np.inf, -np.inf], np.nan).dropna()
    v_series = v_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(c_series) < 2 or len(v_series) < 2:
        return np.nan
    _, p_val = ttest_ind(c_series, v_series, equal_var=False)
    return p_val


def apply_filters(df, filters):
    for col, value in filters.items():
        if col not in df.columns:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            df = df[(df[col] >= value[0]) & (df[col] <= value[1])]
        elif isinstance(value, list):
            df = df[df[col].isin(value)]
        else:
            df = df[df[col] == value]
    return df


# -------------------------------
# Source map
# -------------------------------
source_map = {
    "GAM_MW": ("mw-c", "mw-t"),
    "GAM_DT": ("dt-c", "dt-t"),
    "DIANOMI_MW": (100, 200),
    "DIANOMI_DT": (400, 500)
}


# -------------------------------
# A/B Test Engine
# -------------------------------
def ab_test_engine(df, metrics, source, variant_col="app_name",
                   group_col=None, filters=None, alpha=0.05, date_col="date"):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if filters:
        df = apply_filters(df, filters)

    if source not in source_map:
        raise ValueError(f"Source '{source}' not found in source_map")

    control_name, variant_name = source_map[source]

    # -------------------------------
    # Aggregation
    # -------------------------------
    agg_cols = [date_col, variant_col]
    valid_group_cols = []

    if group_col:
        valid_group_cols = [c for c in group_col if c in df.columns]
        agg_cols += valid_group_cols

    numeric_cols = ["clicks", "impressions", "capacity", "revenue"]
    df_agg = df.groupby(agg_cols)[numeric_cols].sum().reset_index()

    if not valid_group_cols:
        grouped = [("overall", df_agg)]
    else:
        grouped = list(df_agg.groupby(valid_group_cols))

    results = []

    # -------------------------------
    # Metric Computation
    # -------------------------------
    for group_key, temp in grouped:

        min_date = temp[date_col].min()
        max_date = temp[date_col].max()

        group_data = {}
        if not valid_group_cols:
            group_data["group"] = "overall"
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            for col, value in zip(valid_group_cols, group_key):
                group_data[col] = value

        control = temp[temp[variant_col] == control_name]
        variant = temp[temp[variant_col] == variant_name]

        if control.empty or variant.empty:
            continue

        for metric in metrics:

            metric_lower = metric.lower()
            warning_flag = False

            # ---------- CTR (proportion → z-test) ----------
            if metric_lower == "ctr":
                c_clicks = control["clicks"].sum()
                v_clicks = variant["clicks"].sum()
                c_imps = control["impressions"].sum()
                v_imps = variant["impressions"].sum()

                c_val = c_clicks / c_imps if c_imps > 0 else np.nan
                v_val = v_clicks / v_imps if v_imps > 0 else np.nan

                if c_imps > 0 and v_imps > 0:
                    _, p_val = proportions_ztest(
                        [c_clicks, v_clicks],
                        [c_imps, v_imps]
                    )
                else:
                    p_val = np.nan

            # ---------- CPC ----------
            elif metric_lower == "cpc":
                c_val = control["revenue"].sum() / control["clicks"].sum()
                v_val = variant["revenue"].sum() / variant["clicks"].sum()

                c_daily = control["revenue"] / control["clicks"].replace(0, np.nan)
                v_daily = variant["revenue"] / variant["clicks"].replace(0, np.nan)
                p_val = run_ttest(c_daily, v_daily)

            # ---------- CPM ----------
            elif metric_lower == "cpm":
                c_val = (control["revenue"].sum() / control["impressions"].sum()) * 1000
                v_val = (variant["revenue"].sum() / variant["impressions"].sum()) * 1000

                c_daily = (control["revenue"] / control["impressions"].replace(0, np.nan)) * 1000
                v_daily = (variant["revenue"] / variant["impressions"].replace(0, np.nan)) * 1000
                p_val = run_ttest(c_daily, v_daily)

            # ---------- Totals ----------
            elif metric_lower in ["revenue", "clicks", "impressions", "capacity"]:
                c_val = control[metric].sum()
                v_val = variant[metric].sum()
                p_val = run_ttest(control[metric], variant[metric])

            else:
                c_val = v_val = p_val = np.nan
                warning_flag = True

            abs_lift = v_val - c_val
            rel_lift = abs_lift / c_val if c_val != 0 else np.nan

            results.append({
                "source": source,
                "control_name": control_name,
                "variant_name": variant_name,
                **group_data,
                "min_date": min_date,
                "max_date": max_date,
                "metric": metric,
                "control": c_val,
                "variant": v_val,
                "absolute_lift": abs_lift,
                "relative_lift": rel_lift,
                "p_value": p_val,
                "significant": bool(p_val < alpha) if pd.notna(p_val) else False,
                "warning_flag": warning_flag
            })

    result_df = pd.DataFrame(results)
    return result_df


# -------------------------------
# Example Usage
# -------------------------------
results = ab_test_engine(
    DF_GAM,
    source="GAM_MW",
    metrics=["ctr", "revenue", "clicks", "impressions", "capacity", "cpc", "cpm"],
    group_col=["page_type"],
    filters={  "page_type": ["article", "homepage"],
               "page_section": ["news", "xsports"], 
             "date": ("2026-02-01", "2026-02-07") 
    }
)

print(results.to_string(index=False))
