import pandas as pd
import numpy as np
from IPython.display import display

region_map = {
    "NATIONAL CAPITAL REGION (NCR)": "NCR",
    "NATIONAL CAPITAL REGION": "NCR",

    "REGION I (ILOCOS REGION)": "Region I",
    "REGION II (CAGAYAN VALLEY)": "Region II",
    "REGION III (CENTRAL LUZON)": "Region III",
    "REGION IV-A (CALABARZON)": "Region IV-A",
    "REGION IV-B (MIMAROPA)": "Region IV-B",
    "MIMAROPA REGION": "Region IV-B",
    "REGION V (BICOL REGION)": "Region V",
    "REGION VI (WESTERN VISAYAS)": "Region VI",
    "REGION VII (CENTRAL VISAYAS)": "Region VII",
    "REGION VIII (EASTERN VISAYAS)": "Region VIII",
    "REGION IX (ZAMBOANGA PENINSULA)": "Region IX",
    "REGION X (NORTHERN MINDANAO)": "Region X",
    "REGION XI (DAVAO REGION)": "Region XI",
    "REGION XII (SOCCSKSARGEN)": "Region XII",
    "REGION XIII (CARAGA)": "Region XIII",

    "AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)": "ARMM",
    "AUTONOMOUS REGION IN MUSLIM MINDANAO": "ARMM",

    "CORDILLERA ADMINISTRATIVE REGION (CAR)": "CAR",
    "CORDILLERA ADMINISTRATIVE REGION": "CAR",
}

def clean_text(series):
    return (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    )

def normalize_commodity_name(series):
    cleaned = (
        series.astype(str)
        .str.replace('"', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
    )
    cleaned = cleaned.str.upper()
    return cleaned

def normalize_region(series):
    cleaned = (
        series.str.upper()
        .str.strip()
        .replace(region_map)
    )
    cleaned = cleaned.str.upper()
    return cleaned

def unique_non_null_count(values):
    cleaned = pd.Series(values).dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return int(cleaned.nunique())

def filter_commodities(df, invalid_list):
    return df[~df["commodity_name"].isin(invalid_list)].copy()

def transform_series(series, mode):
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if mode == "mom_pct":
        transformed = series.pct_change(fill_method=None) * 100
    elif mode == "diff":
        transformed = series.diff()
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")
    return transformed.replace([np.inf, -np.inf], np.nan)

df_main_1_raw = pd.read_csv("data/main/Philippines - Food Prices.csv")
df_main_2_raw = pd.read_csv("data/main/Openstat Retail Prices.csv")
print("LOADED FOOD AND COMMODITY DATASETS\n")

df_main_1 = df_main_1_raw.copy()
df_main_1["date"] = pd.to_datetime(df_main_1["date"], errors="coerce")
df_main_1["price"] = pd.to_numeric(df_main_1["price"], errors="coerce")
df_main_1 = df_main_1[df_main_1["date"].notna() & df_main_1["price"].notna()].copy()
df_main_1 = df_main_1[df_main_1["pricetype"].astype(str).str.lower() == "retail"].copy()
df_main_1["month"] = df_main_1["date"].dt.to_period("M").dt.to_timestamp()
df_main_1["region"] = normalize_region(clean_text(df_main_1["admin1"]))
df_main_1["island_group"] = (df_main_1["region"])
df_main_1["locality"] = clean_text(df_main_1["admin2"])
df_main_1["market_name"] = clean_text(df_main_1["market"])
df_main_1["commodity_name"] = normalize_commodity_name(clean_text(df_main_1["commodity"]))
df_main_1["source"] = "food_prices"
df_main_1 = df_main_1[
    df_main_1["region"].notna() &
    df_main_1["commodity_name"].notna()
].copy()

df_main_2 = df_main_2_raw.copy()
df_main_2["date"] = pd.to_datetime(df_main_2["date"], errors="coerce")
df_main_2["price"] = pd.to_numeric(df_main_2["price"], errors="coerce")
df_main_2 = df_main_2[df_main_2["date"].notna() & df_main_2["price"].notna()].copy()
df_main_2["month"] = df_main_2["date"].dt.to_period("M").dt.to_timestamp()
df_main_2["region"] = normalize_region(clean_text(df_main_2["region"]))
df_main_2["city_name"] = clean_text(df_main_2["city"])
df_main_2["commodity_name"] = normalize_commodity_name(clean_text(df_main_2["category"]))
df_main_2["source"] = "openstat_retail"
df_main_2 = df_main_2[
    df_main_2["region"].notna() &
    df_main_2["commodity_name"].notna()
].copy()

main_standardized_summary = pd.DataFrame(
    [
        {"dataset": "df_main_1", "rows": len(df_main_1), "unique_regions": df_main_1["region"].nunique(), "unique_commodities": df_main_1["commodity_name"].nunique()},
        {"dataset": "df_main_2", "rows": len(df_main_2), "unique_regions": df_main_2["region"].nunique(), "unique_commodities": df_main_2["commodity_name"].nunique()},
    ]
)
display(main_standardized_summary)

def summarize(df, name):
    summary = (
        df.groupby(["commodity_name", "region"])
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max")
        )
        .reset_index()
    )
    summary["num_months"] = (
            (summary["end_date"].dt.year - summary["start_date"].dt.year) * 12 +
            (summary["end_date"].dt.month - summary["start_date"].dt.month) + 1
    )
    summary["dataset"] = name
    return summary

invalid_commodities = [
    "NEMIMTERID",
    "PALAWAN",
    "PAO GALLIANG",
    "UNKNOWN"
]

df_main_1 = filter_commodities(df_main_1, invalid_commodities)
df_main_2 = filter_commodities(df_main_2, invalid_commodities)

summary_1 = summarize(df_main_1, "df_main_1")
summary_2 = summarize(df_main_2, "df_main_2")
unique_commodity_check = pd.concat([summary_1, summary_2], ignore_index=True)
unique_commodity_check.to_html("tables/1.0.b Commodities x Region.html")

df_main_1_aligned = df_main_1.copy()
df_main_2_aligned = df_main_2.copy()

key_cols = ["month", "region", "commodity_name"]

df1 = (
    df_main_1_aligned
    .groupby(key_cols, as_index=False)["price"]
    .mean()
    .rename(columns={"price": "price_1"})
)

df2 = (
    df_main_2_aligned
    .groupby(key_cols, as_index=False)["price"]
    .mean()
    .rename(columns={"price": "price_2"})
)

merged = df1.merge(df2, on=key_cols, how="outer", indicator=True)

merged["pct_diff"] = (
    (merged["price_1"] - merged["price_2"]).abs()
    / merged[["price_1", "price_2"]].mean(axis=1)
) * 100

merged["price"] = np.where(
    merged["_merge"].eq("both") & (merged["pct_diff"] <= 5),
    merged[["price_1", "price_2"]].mean(axis=1),
    np.where(
        merged["_merge"].eq("left_only"),
        merged["price_1"],
        np.where(
            merged["_merge"].eq("right_only"),
            merged["price_2"],
            np.nan
        )
    )
)
resolved = merged[
    (merged["_merge"] != "both") | (merged["pct_diff"] <= 5)
].copy()

conflicts = merged[
    (merged["_merge"] == "both") & (merged["pct_diff"] > 5)
].copy()

resolved["source"] = np.where(
    resolved["_merge"].eq("left_only"), "food_prices",
    np.where(
        resolved["_merge"].eq("right_only"), "openstat_retail",
        "merged_both"
    )
)
df_combined = resolved[key_cols + ["price", "source"]].copy()

conflicts = merged[
    (merged["_merge"] == "both") & (merged["pct_diff"] > 5)
].copy()

print()
df_combined.info()
print("\nCOMBINED DATASET SUMMARY:")
combined_summary = pd.DataFrame(
    {"commodity": [df_combined["commodity_name"].nunique()], "region": [df_combined["region"].nunique()]}
)
display(combined_summary)

print()
df_combined.to_csv("data/main/Combined Main Dataset.csv", index=False)
conflicts.to_csv("data/main/Combined Main Dataset Conflicts.csv", index=False)
print("EXPORT SUCCESSFUL\nCLEANING AND MERGING MAIN DATASET SUCCESSFUL")
