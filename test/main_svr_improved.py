import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants as const
import helper_functions as hf

from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


TARGET_COL = const.TARGET_COL
TARGET_LABEL = const.TARGET_LABEL
MIN_TRAIN_TARGET_ROWS = const.MIN_TRAIN_TARGET_ROWS
FORECAST_HORIZON = const.FORECAST_HORIZON
SEASONAL_PERIOD = const.SEASONAL_PERIOD
HOLDOUT_RATIO = const.HOLDOUT_RATIO
MIN_TRAIN_OBS_ML = const.MIN_TRAIN_OBS_ML
INNER_FOLDS_ML = const.INNER_FOLDS_ML
MODEL_SERIES_CAP = const.MODEL_SERIES_CAP
MODEL_SERIES_CAP_FALLBACK = const.MODEL_SERIES_CAP_FALLBACK

TABLES_DIR = Path("tables_svr_improved")
VISUALS_DIR = Path("visuals_svr_improved")
TABLES_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)

EXOG_DRIVER_COLUMNS = [
    "diesel_price",
    "food_price_index",
    "max_consecutive_dry_days",
    "days_above_35c",
    "days_rain_above_50mm_extreme",
    "precipitation_mm",
    "avg_mean_temp_c",
]

SVR_IMPROVED_PARAM_GRID = {
    "svr__kernel": ["linear", "rbf"],
    "svr__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    "svr__gamma": ["scale", "auto", 0.01, 0.1],
}


def print_table(title, df):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(empty)")
        return
    print(df.to_string(index=False))


def load_panel():
    df_main = pd.read_csv("data/main/Combined Main Dataset.csv", low_memory=False)
    df_main["month"] = pd.to_datetime(df_main["month"], errors="coerce")
    df_main["price"] = pd.to_numeric(df_main["price"], errors="coerce")

    ex_diesel_raw = pd.read_csv("data/exogenous/Diesel Price.csv")
    ex_food_index_raw = pd.read_csv("data/exogenous/Monthly food price estimates by product and market (2007-2025).csv")
    ex_weather_1 = pd.read_csv("data/exogenous/philippines_weather_cdd_r50mm_hd35_monthly_2000-2023.csv")
    ex_weather_2 = pd.read_csv("data/exogenous/philippines_weather_era5_monthly_2000-2023.csv")

    ex_weather_raw = ex_weather_1.drop(columns=["DaysRainAbove50mm"]).merge(
        ex_weather_2,
        on="Date",
        how="inner",
    )

    ex_diesel_monthly = ex_diesel_raw.copy()
    ex_diesel_monthly["month"] = pd.to_datetime(ex_diesel_monthly["Month"], format="%y-%b", errors="coerce")
    ex_diesel_monthly["month"] = ex_diesel_monthly["month"].dt.to_period("M").dt.to_timestamp()
    ex_diesel_monthly["diesel_price"] = pd.to_numeric(ex_diesel_monthly["Price"], errors="coerce")
    ex_diesel_monthly = (
        ex_diesel_monthly[["month", "diesel_price"]]
        .dropna()
        .sort_values("month")
        .reset_index(drop=True)
    )

    ex_food_index = ex_food_index_raw.copy()
    ex_food_index = ex_food_index[
        ex_food_index["Product"].astype(str).str.strip().str.lower() == "food_price_index"
    ].copy()
    ex_food_index["month"] = pd.to_datetime(ex_food_index["Date"], errors="coerce")
    ex_food_index["month"] = ex_food_index["month"].dt.to_period("M").dt.to_timestamp()
    ex_food_index["food_price_index"] = pd.to_numeric(ex_food_index["Close"], errors="coerce")
    ex_food_index = (
        ex_food_index[["month", "food_price_index"]]
        .dropna()
        .groupby("month", as_index=False)["food_price_index"]
        .mean()
        .sort_values("month")
        .reset_index(drop=True)
    )

    ex_weather_monthly = ex_weather_raw.copy()
    ex_weather_monthly["month"] = pd.to_datetime(ex_weather_monthly["Date"], errors="coerce")
    ex_weather_monthly = ex_weather_monthly.rename(
        columns={
            "MaxConsecutiveDryDays": "max_consecutive_dry_days",
            "DaysMaxTempAbove35C": "days_above_35c",
            "DaysRainAbove50mm": "days_rain_above_50mm_extreme",
            "Precipitation_mm": "precipitation_mm",
            "AvgMeanTemp_C": "avg_mean_temp_c",
        }
    )
    ex_weather_monthly = ex_weather_monthly.drop(columns=["Date"])

    macro_external_monthly = (
        ex_diesel_monthly
        .merge(ex_food_index, on="month", how="outer")
        .merge(ex_weather_monthly, on="month", how="outer")
        .sort_values("month")
        .groupby("month", as_index=False).mean(numeric_only=True)
        .reset_index(drop=True)
    )
    macro_external_monthly = macro_external_monthly.dropna(
        subset=macro_external_monthly.columns.difference(["month"])
    )

    valid_series = []
    for (region, commodity_name), group in df_main.groupby(["region", "commodity_name"]):
        interpolated = hf.apply_linear_interpolation(group, max_gap=2)
        if interpolated is None:
            continue
        if len(interpolated) >= 36:
            interpolated["region"] = region
            interpolated["commodity_name"] = commodity_name
            valid_series.append(interpolated)

    df_main_filtered = pd.concat(valid_series, ignore_index=True)
    df_main_filtered = df_main_filtered.sort_values(
        ["region", "commodity_name", "month"]
    ).reset_index(drop=True)

    panel = df_main_filtered.merge(macro_external_monthly, on="month", how="left")
    panel = hf.add_basic_price_features(panel)
    panel, exog_feature_columns = hf.add_exogenous_lags(panel, drivers=EXOG_DRIVER_COLUMNS)

    manifest, panel = hf.build_series_manifest(panel)
    if len(manifest) >= MODEL_SERIES_CAP:
        modeling_series_cap = MODEL_SERIES_CAP
    elif len(manifest) >= MODEL_SERIES_CAP_FALLBACK:
        modeling_series_cap = MODEL_SERIES_CAP_FALLBACK
    else:
        modeling_series_cap = len(manifest)

    eligible_series_manifest = (
        manifest.sort_values(
            ["months_total", "yoy_rows", "region", "commodity_name"],
            ascending=[False, False, True, True],
        )
        .head(modeling_series_cap)
        .reset_index(drop=True)
    )
    eligible_ids = set(eligible_series_manifest["series_id"])
    eligible_panel = panel.loc[panel["series_id"].isin(eligible_ids)].copy()

    return eligible_panel, eligible_series_manifest, exog_feature_columns


def choose_svr_features(panel, exog_feature_columns):
    base_features = [
        "yoy_lag_1",
        "yoy_lag_2",
        "yoy_lag_3",
        "yoy_lag_6",
        "yoy_lag_12",
        "yoy_roll_mean_3",
        "yoy_roll_mean_6",
        "yoy_roll_std_3",
        "yoy_roll_std_6",
        "yoy_acceleration",
        "price_index_lag_1",
        "price_index_lag_3",
        "price_index_roll_mean_3",
        "price_index_roll_std_3",
        "log_price_lag_1",
        "log_price_lag_3",
        "yoy_lag1_x_roll3",
        "yoy_lag1_x_roll6",
        "month_sin",
        "month_cos",
    ]
    base_features = [feature for feature in base_features if feature in panel.columns]

    exog_candidates = [feature for feature in exog_feature_columns if feature in panel.columns]
    if exog_candidates:
        exog_coverage = panel[exog_candidates].notna().mean().sort_values(ascending=False)
        exog_features = exog_coverage.loc[exog_coverage >= 0.85].index.tolist()[:3]
    else:
        exog_features = []

    return base_features + exog_features


def build_svr_residual_frame(df, feature_cols):
    df = df.sort_values("month").reset_index(drop=True).copy()
    df["seasonal_baseline_feature"] = df["yoy_lag_12"].combine_first(df["yoy_lag_1"])

    valid_positions = df.index[df[TARGET_COL].notna()].tolist()
    if len(valid_positions) < MIN_TRAIN_OBS_ML + 1:
        return None

    holdout_count = max(1, int(np.ceil(len(valid_positions) * HOLDOUT_RATIO)))
    holdout_count = min(holdout_count, len(valid_positions) - MIN_TRAIN_OBS_ML)
    if holdout_count <= 0:
        return None

    test_positions = valid_positions[-holdout_count:]
    first_test_position = test_positions[0]
    first_train = df.iloc[:first_test_position].copy()

    tuning = first_train.loc[
        first_train[feature_cols + [TARGET_COL, "seasonal_baseline_feature"]].notna().all(axis=1),
        ["month"] + feature_cols + [TARGET_COL, "seasonal_baseline_feature"],
    ].copy().reset_index(drop=True)
    if len(tuning) < max(18, min(30, len(feature_cols) + 8)):
        return None

    tuning["residual_target"] = tuning[TARGET_COL] - tuning["seasonal_baseline_feature"]
    n_splits = min(INNER_FOLDS_ML, max(2, len(tuning) - MIN_TRAIN_OBS_ML))
    if n_splits < 2:
        return None

    return {
        "test_positions": test_positions,
        "tuning": tuning,
        "time_cv": TimeSeriesSplit(n_splits=n_splits),
    }


def select_best_svr_model(tuning, time_cv, feature_cols):
    X = tuning[feature_cols]
    baseline = tuning["seasonal_baseline_feature"].to_numpy(dtype=float)
    y = tuning[TARGET_COL].to_numpy(dtype=float)
    residual_y = tuning["residual_target"].to_numpy(dtype=float)

    baseline_preds = []
    baseline_actuals = []
    for _, val_idx in time_cv.split(X):
        baseline_preds.extend(baseline[val_idx].tolist())
        baseline_actuals.extend(y[val_idx].tolist())
    baseline_rmse = hf.rmse(baseline_actuals, baseline_preds)

    best_params = None
    best_rmse = float("inf")

    for params in ParameterGrid(SVR_IMPROVED_PARAM_GRID):
        if params["svr__kernel"] == "linear" and params["svr__gamma"] not in {"scale", "auto"}:
            continue

        cv_preds = []
        cv_actuals = []

        for train_idx, val_idx in time_cv.split(X):
            X_fit = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_fit = residual_y[train_idx]
            y_val = y[val_idx]
            baseline_val = baseline[val_idx]

            model = Pipeline([
                ("scaler", RobustScaler()),
                ("svr", SVR(
                    kernel=params["svr__kernel"],
                    C=params["svr__C"],
                    epsilon=params["svr__epsilon"],
                    gamma=params["svr__gamma"],
                )),
            ])

            try:
                model.fit(X_fit, y_fit)
                residual_pred = model.predict(X_val)
                final_pred = baseline_val + residual_pred
            except Exception:
                final_pred = baseline_val

            cv_preds.extend(final_pred.tolist())
            cv_actuals.extend(y_val.tolist())

        current_rmse = hf.rmse(cv_actuals, cv_preds)
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params

    return best_params, best_rmse, baseline_rmse


def run_svr_models_improved(manifest, panel, feature_cols):
    prediction_rows = []
    settings_rows = []

    for _, meta in manifest.iterrows():
        df = panel.loc[panel["series_id"] == meta["series_id"]].sort_values("month").reset_index(drop=True)
        prepared = build_svr_residual_frame(df, feature_cols)
        if prepared is None:
            continue

        test_positions = prepared["test_positions"]
        tuning = prepared["tuning"]
        time_cv = prepared["time_cv"]

        best_params, best_rmse, baseline_rmse = select_best_svr_model(tuning, time_cv, feature_cols)
        if best_params is None:
            continue

        use_residual_model = best_rmse < baseline_rmse

        settings_rows.append({
            "series_id": meta["series_id"],
            "region": meta["region"],
            "commodity_name": meta["commodity_name"],
            "selected_features": str(feature_cols),
            "svr_best_params": str(best_params),
            "svr_inner_rmse": best_rmse,
            "seasonal_baseline_inner_rmse": baseline_rmse,
            "uses_residual_model": use_residual_model,
            "holdout_rows": len(test_positions),
        })

        for holdout_position in test_positions:
            train_block = df.iloc[:holdout_position].copy()
            row = df.iloc[holdout_position].copy()
            actual = row[TARGET_COL]
            if pd.isna(actual):
                continue

            history = train_block.loc[train_block[TARGET_COL].notna(), ["month", TARGET_COL]].copy().reset_index(drop=True)
            if history.empty:
                continue

            naive_pred = float(history[TARGET_COL].iloc[-1])
            seasonal_pred = hf.seasonal_naive_forecast(history, row["month"], TARGET_COL)
            if pd.isna(seasonal_pred):
                seasonal_pred = naive_pred

            train_block["seasonal_baseline_feature"] = train_block["yoy_lag_12"].combine_first(train_block["yoy_lag_1"])
            model_train = train_block.loc[
                train_block[feature_cols + [TARGET_COL, "seasonal_baseline_feature"]].notna().all(axis=1),
                feature_cols + [TARGET_COL, "seasonal_baseline_feature"],
            ].copy()
            model_train["residual_target"] = model_train[TARGET_COL] - model_train["seasonal_baseline_feature"]

            feature_ready = bool(feature_cols) and row[feature_cols].notna().all()
            if use_residual_model and len(model_train) >= MIN_TRAIN_OBS_ML and feature_ready:
                X_train = model_train[feature_cols]
                y_train = model_train["residual_target"]
                X_test = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
                try:
                    final_model = Pipeline([
                        ("scaler", RobustScaler()),
                        ("svr", SVR(
                            kernel=best_params["svr__kernel"],
                            C=best_params["svr__C"],
                            epsilon=best_params["svr__epsilon"],
                            gamma=best_params["svr__gamma"],
                        )),
                    ])
                    final_model.fit(X_train, y_train)
                    residual_pred = float(final_model.predict(X_test)[0])
                    svr_pred = float(seasonal_pred + residual_pred)
                except Exception:
                    svr_pred = float(seasonal_pred)
            else:
                svr_pred = float(seasonal_pred)

            prediction_rows.append({
                "series_id": meta["series_id"],
                "region": meta["region"],
                "commodity_name": meta["commodity_name"],
                "month": row["month"],
                "actual": float(actual),
                "naive_pred": float(naive_pred),
                "seasonal_naive_pred": float(seasonal_pred),
                "svr_pred": float(svr_pred),
            })

    prediction_columns = [
        "series_id",
        "region",
        "commodity_name",
        "month",
        "actual",
        "naive_pred",
        "seasonal_naive_pred",
        "svr_pred",
    ]
    settings_columns = [
        "series_id",
        "region",
        "commodity_name",
        "selected_features",
        "svr_best_params",
        "svr_inner_rmse",
        "seasonal_baseline_inner_rmse",
        "uses_residual_model",
        "holdout_rows",
    ]
    return (
        pd.DataFrame(prediction_rows, columns=prediction_columns),
        pd.DataFrame(settings_rows, columns=settings_columns),
    )


def main():
    eligible_panel, eligible_series_manifest, regional_exogenous_feature_columns = load_panel()
    svr_feature_cols = choose_svr_features(eligible_panel, regional_exogenous_feature_columns)

    target_ready_panel = eligible_panel.loc[eligible_panel[TARGET_COL].notna()].copy()
    split_readiness = (
        target_ready_panel.groupby("series_id")
        .agg(
            region=("region", "first"),
            commodity_name=("commodity_name", "first"),
            target_rows=(TARGET_COL, "size"),
        )
        .reset_index()
    )
    split_readiness["walk_forward_ready"] = split_readiness["target_rows"] >= (MIN_TRAIN_TARGET_ROWS + FORECAST_HORIZON)

    model_manifest = eligible_series_manifest.merge(
        split_readiness[["series_id", "walk_forward_ready"]],
        on="series_id",
        how="left",
    )
    model_manifest = model_manifest.loc[model_manifest["walk_forward_ready"].fillna(False)].copy()

    feature_manifest = pd.DataFrame([
        {
            "candidate_series": int(len(model_manifest)),
            "svr_feature_count": int(len(svr_feature_cols)),
            "svr_features": ", ".join(svr_feature_cols),
            "strategy": "seasonal-naive residual learning with RobustScaler",
        }
    ])
    feature_manifest.to_html(TABLES_DIR / "1.1.a SVR Improved Feature Manifest.html", index=False)

    svr_predictions, svr_model_settings = run_svr_models_improved(
        model_manifest,
        eligible_panel,
        svr_feature_cols,
    )

    run_overview = pd.DataFrame([
        {
            "prediction_rows": int(len(svr_predictions)),
            "svr_models": int(len(svr_model_settings)),
            "series_using_residual_model": int(svr_model_settings["uses_residual_model"].fillna(False).sum()) if not svr_model_settings.empty else 0,
            "svr_feature_count": int(len(svr_feature_cols)),
        }
    ])

    prediction_pairs = [
        ("Naive", "naive_pred"),
        ("Seasonal Naive", "seasonal_naive_pred"),
        ("SVR Improved", "svr_pred"),
    ]

    svr_global_metrics = hf.compute_metrics_table(svr_predictions, prediction_pairs)
    svr_series_metrics = hf.compute_series_metrics(svr_predictions, prediction_pairs)
    svr_diagnostics = hf.compute_diagnostics(svr_predictions, prediction_pairs)

    feature_manifest.to_html(TABLES_DIR / "1.1.a SVR Improved Feature Manifest.html", index=False)
    run_overview.to_html(TABLES_DIR / "1.1.b SVR Improved Run Overview.html", index=False)
    svr_model_settings.to_html(TABLES_DIR / "1.1.c SVR Improved Model Settings.html", index=False)
    svr_predictions.to_html(TABLES_DIR / "1.1.d SVR Improved Predictions.html", index=False)
    svr_global_metrics.to_html(TABLES_DIR / "1.1.e SVR Improved Global Metrics.html", index=False)
    svr_series_metrics.to_html(TABLES_DIR / "1.1.f SVR Improved Series Metrics.html", index=False)
    svr_diagnostics.to_html(TABLES_DIR / "1.1.g SVR Improved Diagnostics.html", index=False)

    print_table("SVR Improved Run Overview", run_overview)
    if svr_predictions.empty:
        print("\nNo SVR improved predictions were generated. The residual-model gate likely rejected all series or feature completeness was insufficient.")
    print_table("SVR Improved Global Metrics", svr_global_metrics)

    if not svr_global_metrics.empty:
        benchmark_plot = svr_global_metrics.set_index("model").loc[
            ["Naive", "Seasonal Naive", "SVR Improved"]
        ].reset_index()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(benchmark_plot["model"], benchmark_plot["rmse"])
        ax.set_title("SVR Improved vs Baselines")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plt.savefig(VISUALS_DIR / "1.1.a SVR Improved Benchmark RMSE Comparison.png", dpi=200, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
