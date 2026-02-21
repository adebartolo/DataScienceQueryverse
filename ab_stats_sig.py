# A/B Testing for determining Lift and Stats Sig
# Handles multiple metrics, multi-level grouping, and optional filters.
# Differentiates between proportion metrics (CTR → z-test) and ratio/continuous metrics (CPC, CPM → t-test).
# Outputs a clean, ordered table with lifts, p-values, and flags for warnings or significance

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------
# Step 1 - Fake data
# -------------------------------

# Simulate daily aggregated data

np.random.seed(42)
days = pd.date_range(start="2026-02-01", periods=14)

slots = ["banner", "multi1", "multi2", "multi3"]
page_types = ["article", "homepage"]
page_sections = ["news", "sports", "finance"]
vendors = ["GAM"]

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
                        "vendor": vendors[0],
                        "clicks": clicks,
                        "impressions": impressions,
                        "capacity": capacity,
                        "revenue": revenue
                    })

DF_GAM = pd.DataFrame(rows)

# Derived metrics
DF_GAM["ctr"] = DF_GAM["clicks"] / DF_GAM["impressions"]

print("Simulated DF_GAM:")
print(DF_GAM.head())

# -------------------------------
# Step 2 - Define how to get lift and stats sig
# -------------------------------

# A/B Test Engine

# A/B Testing for determining Lift and Stats Sig
# Handles multiple metrics, multi-level grouping, and optional filters.
# Differentiates between proportion metrics (CTR → z-test) and ratio/continuous metrics (CPC, CPM → t-test).
# Outputs a clean, ordered table with lifts, p-values, and flags for warnings or significance

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

# -------------------------------
# Source Map
# -------------------------------
source_map = {
    "GAM_MW": ("mw-c", "mw-t"), # control, variant
    "GAM_DT": ("dt-c", "dt-t"),
    "DIANOMI_MW": (100, 200),
    "DIANOMI_DT": (400, 500)
}

# -------------------------------
# Helper Functions
# -------------------------------
def safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else np.nan


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

        # Range filter
        if isinstance(value, tuple) and len(value) == 2:
            df = df[(df[col] >= value[0]) & (df[col] <= value[1])]

        # Multi-value filter
        elif isinstance(value, list):
            df = df[df[col].isin(value)]

        # Exact match
        else:
            df = df[df[col] == value]

    return df


# -------------------------------
# A/B Test Engine
# -------------------------------
def ab_test_engine(
    df,
    metrics,
    source,
    variant_col="app_name",
    group_col=None,
    filters=None,
    alpha=0.05,
    date_col="date"
):

    df = df.copy()
    # -------------------------------
    # Ensure date column is datetime
    # -------------------------------
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # -------------------------------
    # Apply Optional Filters
    # -------------------------------
    if filters:
        df = apply_filters(df, filters)

    # -------------------------------
    # Validate Source
    # -------------------------------
    if source not in source_map:
        raise ValueError(f"Source '{source}' not found in source_map")

    control_name, variant_name = source_map[source]

    # -------------------------------
    # Handle Grouping
    # -------------------------------
    if group_col is None:
        grouped = [("overall", df)]
    else:
        if isinstance(group_col, str):
            group_col = [group_col]
        grouped = list(df.groupby(group_col))

    results = []

    for group_key, temp in grouped:

        # -------------------------------
        # Date range calculation
        # -------------------------------
        if date_col in temp.columns:
            min_date = temp[date_col].min()
            max_date = temp[date_col].max()
        else:
            min_date = None
            max_date = None

        # -------------------------------
        # Prepare grouping columns
        # -------------------------------
        group_data = {}

        if group_col is None:
            group_data["group"] = "overall"
        else:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            for col, value in zip(group_col, group_key):
                group_data[col] = value

        control = temp[temp[variant_col] == control_name]
        variant = temp[temp[variant_col] == variant_name]

        if control.empty or variant.empty:
            continue

        for metric in metrics:

            metric_lower = metric.lower()
            warning_flag = False

            # CTR
            if metric_lower == "ctr":

                c_clicks = control["clicks"].sum()
                v_clicks = variant["clicks"].sum()
                c_impr = control["impressions"].sum()
                v_impr = variant["impressions"].sum()

                c_val = safe_div(c_clicks, c_impr)
                v_val = safe_div(v_clicks, v_impr)

                if c_impr > 0 and v_impr > 0:
                    p_val = proportions_ztest(
                        [c_clicks, v_clicks],
                        [c_impr, v_impr]
                    )[1]
                else:
                    p_val = np.nan

            elif metric_lower == "cpc":

                c_val = safe_div(control["revenue"].sum(), control["clicks"].sum())
                v_val = safe_div(variant["revenue"].sum(), variant["clicks"].sum())

                p_val = run_ttest(
                    control["revenue"] / control["clicks"],
                    variant["revenue"] / variant["clicks"]
                )

            elif metric_lower == "cpm":

                c_val = safe_div(control["revenue"].sum(), control["impressions"].sum()) * 1000
                v_val = safe_div(variant["revenue"].sum(), variant["impressions"].sum()) * 1000

                p_val = run_ttest(
                    control["revenue"] / control["impressions"] * 1000,
                    variant["revenue"] / variant["impressions"] * 1000
                )

            else:

                if metric not in df.columns:
                    c_val = v_val = p_val = np.nan
                    warning_flag = True
                else:
                    c_series = control[metric].dropna()
                    v_series = variant[metric].dropna()

                    c_val = c_series.mean() if not c_series.empty else np.nan
                    v_val = v_series.mean() if not v_series.empty else np.nan
                    p_val = run_ttest(c_series, v_series)

            abs_lift = v_val - c_val if pd.notna(c_val) else np.nan
            rel_lift = safe_div(abs_lift, c_val)

            result_row = {
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
                "significant": p_val < alpha if pd.notna(p_val) else False,
                "warning_flag": warning_flag or pd.isna(c_val) or pd.isna(v_val)
            }

            results.append(result_row)

    result_df = pd.DataFrame(results)

    # -------------------------------
    # Column Ordering
    # -------------------------------
    base_cols = ["source", "control_name", "variant_name"]
    date_cols = ["min_date", "max_date"]

    if group_col is None:
        group_cols = ["group"]
    else:
        group_cols = group_col if isinstance(group_col, list) else [group_col]

    metric_cols = [
        "metric",
        "control",
        "variant",
        "absolute_lift",
        "relative_lift",
        "p_value",
        "significant",
        "warning_flag"
    ]

    ordered_cols = date_cols + base_cols + group_cols + metric_cols
    ordered_cols = [col for col in ordered_cols if col in result_df.columns]

    return result_df[ordered_cols]

# -------------------------------
# Step 3 - Call function
# -------------------------------

results = ab_test_engine(
    DF_GAM,
    source="GAM_MW",
    metrics=["ctr", "revenue", "clicks", "impressions", "capacity", "cpc", "cpm"]
    #,group_col=["page_type","page_section"]
    ,filters={
        # "date": ("2026-02-01", "2026-02-14")
        # ,"page_section": ["sports"]
        # ,"page_type": ["article"]
    }
)

print(results)
