import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX

import constants as const
import helper_functions as hf
import webapp_export as web_export


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reviewer" / "region_vi_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REGION = "REGION VI"
TARGET_SERIES_COUNT = 5
RUN_COUNT = 10
APP_MODELS = ["SARIMA", "SVR", "LightGBM", "Weighted Ensemble"]
MODEL_COLUMN_MAP = {
    "SARIMA": "sarima_pred",
    "SVR": "svr_pred",
    "LightGBM": "lightgbm_pred",
    "Weighted Ensemble": "weighted_ensemble_pred",
}
DEFAULT_REGION_VI_COMMODITIES = [
    "AVOCADO",
    "BAGUIO BEANS",
    "BEEF",
    "BANANA",
    "ALOY/TULINGAN",
]

TARGET_COL = const.TARGET_COL
TARGET_LABEL = const.TARGET_LABEL
MIN_TRAIN_TARGET_ROWS = const.MIN_TRAIN_TARGET_ROWS
FORECAST_HORIZON = const.FORECAST_HORIZON
HOLDOUT_RATIO = const.HOLDOUT_RATIO
MIN_TRAIN_OBS_SARIMA = const.MIN_TRAIN_OBS_SARIMA
MIN_TRAIN_OBS_ML = const.MIN_TRAIN_OBS_ML
SEASONAL_PERIOD = const.SEASONAL_PERIOD
SARIMA_MAXITER = const.SARIMA_MAXITER
LIGHTGBM_RANDOM_STATE = const.LIGHTGBM_RANDOM_STATE
SVR_BASE_FEATURES = const.SVR_BASE_FEATURES
LIGHTGBM_BASE_FEATURES = const.LIGHTGBM_BASE_FEATURES

N_JOBS, BACKEND = hf.get_environment_config()


def compute_split_readiness(panel):
    target_ready_panel = panel.loc[panel[TARGET_COL].notna()].copy()
    split_readiness = (
        target_ready_panel.groupby("series_id")
        .agg(
            region=("region", "first"),
            commodity_name=("commodity_name", "first"),
            target_rows=(TARGET_COL, "size"),
            first_target_month=("month", "min"),
            last_target_month=("month", "max"),
        )
        .reset_index()
    )
    split_readiness["walk_forward_ready"] = (
        split_readiness["target_rows"] >= (MIN_TRAIN_TARGET_ROWS + FORECAST_HORIZON)
    )
    return split_readiness


def compute_stationarity_results(panel):
    rows = []
    for series_id, part in panel.groupby("series_id"):
        target_series = part[TARGET_COL].dropna()
        adf_p_value_aic = hf.safe_adf_pvalue(target_series, autolag="AIC")
        adf_p_value_bic = hf.safe_adf_pvalue(target_series, autolag="BIC")

        recommended_d_aic = 0 if pd.notna(adf_p_value_aic) and adf_p_value_aic < 0.05 else 1 if pd.notna(adf_p_value_aic) else np.nan
        recommended_d_bic = 0 if pd.notna(adf_p_value_bic) and adf_p_value_bic < 0.05 else 1 if pd.notna(adf_p_value_bic) else np.nan

        if pd.notna(recommended_d_aic) and pd.notna(recommended_d_bic):
            recommended_d = int(max(recommended_d_aic, recommended_d_bic))
        elif pd.notna(recommended_d_aic):
            recommended_d = int(recommended_d_aic)
        elif pd.notna(recommended_d_bic):
            recommended_d = int(recommended_d_bic)
        else:
            recommended_d = np.nan

        rows.append({
            "series_id": series_id,
            "region": part["region"].iloc[0],
            "commodity_name": part["commodity_name"].iloc[0],
            "target_rows": int(target_series.shape[0]),
            "adf_p_value_aic": adf_p_value_aic,
            "adf_p_value_bic": adf_p_value_bic,
            "recommended_d": recommended_d,
        })
    return pd.DataFrame(rows)


def compute_seasonality_results(panel):
    rows = []
    for series_id, part in panel.groupby("series_id"):
        target_series = part[TARGET_COL]
        lag12_autocorr = hf.lagged_autocorr(target_series, SEASONAL_PERIOD)
        month_profile = part.groupby("month_num")[TARGET_COL].mean().reindex(range(1, 13))
        month_profile_std = month_profile.std() if not month_profile.empty else np.nan
        recommended_D = 1 if pd.notna(lag12_autocorr) and abs(lag12_autocorr) >= 0.20 else 0 if pd.notna(lag12_autocorr) else np.nan

        rows.append({
            "series_id": series_id,
            "region": part["region"].iloc[0],
            "commodity_name": part["commodity_name"].iloc[0],
            "lag12_autocorr": lag12_autocorr,
            "month_profile_std": month_profile_std,
            "recommended_D": recommended_D,
        })
    return pd.DataFrame(rows)


def build_sarima_readiness(manifest, panel):
    split_readiness = compute_split_readiness(panel)
    stationarity_results = compute_stationarity_results(panel)
    seasonality_results = compute_seasonality_results(panel)
    sarima_readiness = (
        manifest[
            ["series_id", "region", "commodity_name", "months_total", "price_rows", "mom_rows", "yoy_rows"]
        ]
        .merge(
            split_readiness[["series_id", "target_rows", "walk_forward_ready"]],
            on="series_id",
            how="left",
        )
        .merge(
            stationarity_results[["series_id", "adf_p_value_aic", "adf_p_value_bic", "recommended_d"]],
            on="series_id",
            how="left",
        )
        .merge(
            seasonality_results[["series_id", "lag12_autocorr", "recommended_D", "month_profile_std"]],
            on="series_id",
            how="left",
        )
        .sort_values(["commodity_name", "region"])
        .reset_index(drop=True)
    )
    return sarima_readiness


def load_region_vi_dashboard_preferences():
    dashboard_path = ROOT / "webapp" / "data" / "dashboard.json"
    if not dashboard_path.exists():
        return []

    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("series_metrics", []) if row.get("region") == TARGET_REGION]
    if not rows:
        return []

    summary_rows = []
    for series_id, part in pd.DataFrame(rows).groupby("series_id"):
        model_lookup = {row["model"]: row for _, row in part.iterrows()}
        non_negative_count = int(
            sum(
                1
                for model_name in APP_MODELS
                if pd.notna(model_lookup.get(model_name, {}).get("r2"))
                and float(model_lookup[model_name]["r2"]) >= 0
            )
        )
        summary_rows.append({
            "series_id": series_id,
            "commodity_name": part["commodity_name"].iloc[0],
            "non_negative_model_count": non_negative_count,
            "ensemble_r2": model_lookup.get("Weighted Ensemble", {}).get("r2", -np.inf),
            "mean_r2": float(pd.to_numeric(part["r2"], errors="coerce").mean()),
        })

    preferred = pd.DataFrame(summary_rows)
    preferred["ensemble_r2"] = pd.to_numeric(preferred["ensemble_r2"], errors="coerce")
    preferred = preferred.sort_values(
        ["non_negative_model_count", "ensemble_r2", "mean_r2", "commodity_name"],
        ascending=[False, False, False, True],
    )
    return preferred["commodity_name"].tolist()


def select_focus_manifest(manifest):
    region_manifest = manifest.loc[manifest["region"] == TARGET_REGION].copy()
    if region_manifest.empty:
        raise ValueError(f"No eligible series found for {TARGET_REGION}.")

    preferred_commodities = []
    for commodity in load_region_vi_dashboard_preferences() + DEFAULT_REGION_VI_COMMODITIES:
        if commodity not in preferred_commodities:
            preferred_commodities.append(commodity)

    selected_rows = []
    selected_series_ids = set()

    for commodity in preferred_commodities:
        commodity_rows = region_manifest.loc[region_manifest["commodity_name"] == commodity].copy()
        if commodity_rows.empty:
            continue
        commodity_rows = commodity_rows.sort_values(["months_total", "yoy_rows"], ascending=[False, False])
        row = commodity_rows.iloc[0]
        if row["series_id"] in selected_series_ids:
            continue
        selected_rows.append(row)
        selected_series_ids.add(row["series_id"])
        if len(selected_rows) >= TARGET_SERIES_COUNT:
            break

    if len(selected_rows) < TARGET_SERIES_COUNT:
        for _, row in region_manifest.sort_values(["months_total", "yoy_rows", "commodity_name"], ascending=[False, False, True]).iterrows():
            if row["series_id"] in selected_series_ids:
                continue
            selected_rows.append(row)
            selected_series_ids.add(row["series_id"])
            if len(selected_rows) >= TARGET_SERIES_COUNT:
                break

    selected_manifest = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected_manifest


def build_component_prediction_frame(series_id, model_name, months, actuals, preds):
    return pd.DataFrame({
        "series_id": series_id,
        "month": months,
        "actual": actuals,
        MODEL_COLUMN_MAP[model_name]: preds,
    })


def get_holdout_partition(df, min_train_obs):
    valid_positions = df.index[df[TARGET_COL].notna()].tolist()
    if len(valid_positions) < min_train_obs + 1:
        return None

    holdout_count = max(1, int(np.ceil(len(valid_positions) * HOLDOUT_RATIO)))
    holdout_count = min(holdout_count, len(valid_positions) - min_train_obs)
    if holdout_count <= 0:
        return None

    test_positions = valid_positions[-holdout_count:]
    first_test_position = test_positions[0]
    return {
        "test_positions": test_positions,
        "first_train": df.iloc[:first_test_position].copy(),
    }


def compute_sarima_train_predictions(panel, settings_df):
    rows = []
    for _, row in settings_df.iterrows():
        series_id = row["series_id"]
        df = panel.loc[panel["series_id"] == series_id].sort_values("month").reset_index(drop=True).copy()
        partition = get_holdout_partition(df, MIN_TRAIN_OBS_SARIMA)
        if partition is None:
            continue

        tuning = partition["first_train"].loc[
            partition["first_train"][TARGET_COL].notna(),
            ["month", TARGET_COL],
        ].copy().reset_index(drop=True)
        if len(tuning) < MIN_TRAIN_OBS_SARIMA:
            continue

        order = hf.parse_literal_or_default(row.get("sarima_order"), fallback=None)
        seasonal_order = hf.parse_literal_or_default(row.get("sarima_seasonal_order"), fallback=None)
        trend = row.get("sarima_trend", "n")
        if order is None or seasonal_order is None:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    tuning[TARGET_COL],
                    order=tuple(order),
                    seasonal_order=tuple(seasonal_order),
                    trend=trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=SARIMA_MAXITER)
            predictions = pd.Series(np.asarray(model.fittedvalues, dtype=float), index=tuning.index)
        except Exception:
            continue

        valid = pd.DataFrame({
            "series_id": series_id,
            "month": tuning["month"],
            "actual": tuning[TARGET_COL],
            MODEL_COLUMN_MAP["SARIMA"]: predictions,
        }).dropna(subset=["actual", MODEL_COLUMN_MAP["SARIMA"]])
        rows.append(valid)

    if not rows:
        return pd.DataFrame(columns=["series_id", "month", "actual", MODEL_COLUMN_MAP["SARIMA"]])
    return pd.concat(rows, ignore_index=True)


def compute_svr_train_predictions(panel, settings_df):
    rows = []
    for _, row in settings_df.iterrows():
        series_id = row["series_id"]
        df = panel.loc[panel["series_id"] == series_id].sort_values("month").reset_index(drop=True).copy()
        partition = get_holdout_partition(df, MIN_TRAIN_OBS_ML)
        if partition is None:
            continue

        feature_cols = hf.parse_literal_or_default(row.get("selected_features"), fallback=[]) or []
        best_params = hf.parse_literal_or_default(row.get("svr_best_params"), fallback={}) or {}
        if not feature_cols:
            continue

        model_train = partition["first_train"].loc[
            partition["first_train"][feature_cols + [TARGET_COL]].notna().all(axis=1),
            ["month"] + feature_cols + [TARGET_COL],
        ].copy()
        if len(model_train) < MIN_TRAIN_OBS_ML:
            continue

        try:
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("svr", SVR(
                    kernel="rbf",
                    C=best_params.get("svr__C", 1.0),
                    epsilon=best_params.get("svr__epsilon", 0.1),
                    gamma=best_params.get("svr__gamma", "scale"),
                )),
            ])
            model.fit(model_train[feature_cols], model_train[TARGET_COL])
            predictions = model.predict(model_train[feature_cols])
        except Exception:
            continue

        rows.append(
            build_component_prediction_frame(
                series_id,
                "SVR",
                model_train["month"].tolist(),
                model_train[TARGET_COL].tolist(),
                predictions,
            )
        )

    if not rows:
        return pd.DataFrame(columns=["series_id", "month", "actual", MODEL_COLUMN_MAP["SVR"]])
    return pd.concat(rows, ignore_index=True)


def compute_lightgbm_train_predictions(panel, settings_df):
    rows = []
    for _, row in settings_df.iterrows():
        series_id = row["series_id"]
        df = panel.loc[panel["series_id"] == series_id].sort_values("month").reset_index(drop=True).copy()
        partition = get_holdout_partition(df, MIN_TRAIN_OBS_ML)
        if partition is None:
            continue

        feature_cols = hf.parse_literal_or_default(row.get("selected_features"), fallback=[]) or []
        exog_cols = hf.parse_literal_or_default(row.get("lightgbm_exogenous_features"), fallback=[]) or []
        best_params = hf.parse_literal_or_default(row.get("lightgbm_params"), fallback={}) or {}
        if not feature_cols:
            continue

        model_train = partition["first_train"].copy()
        for col in exog_cols:
            if col in model_train.columns:
                model_train[col] = model_train[col].ffill().bfill()

        model_train = model_train.loc[
            model_train[feature_cols + [TARGET_COL]].notna().all(axis=1),
            ["month"] + feature_cols + [TARGET_COL],
        ].copy()
        if len(model_train) < MIN_TRAIN_OBS_ML:
            continue

        try:
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=LIGHTGBM_RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,
                force_col_wise=True,
                max_bin=127,
                **best_params,
            )
            model.fit(model_train[feature_cols], model_train[TARGET_COL])
            predictions = model.predict(model_train[feature_cols])
        except Exception:
            continue

        rows.append(
            build_component_prediction_frame(
                series_id,
                "LightGBM",
                model_train["month"].tolist(),
                model_train[TARGET_COL].tolist(),
                predictions,
            )
        )

    if not rows:
        return pd.DataFrame(columns=["series_id", "month", "actual", MODEL_COLUMN_MAP["LightGBM"]])
    return pd.concat(rows, ignore_index=True)


def combine_component_predictions(sarima_df, svr_df, lightgbm_df, ensemble_weights):
    merged = None
    for frame in [sarima_df, svr_df, lightgbm_df]:
        if frame.empty:
            continue
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on=["series_id", "month", "actual"], how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["series_id", "month", "actual", "weighted_ensemble_pred"])

    merged = merged.merge(
        ensemble_weights[["series_id", "sarima_inner_rmse", "svr_inner_rmse", "lightgbm_inner_rmse"]],
        on="series_id",
        how="left",
    )

    weighted_preds = []
    for _, row in merged.iterrows():
        preds = np.asarray(
            [row.get("sarima_pred"), row.get("svr_pred"), row.get("lightgbm_pred")],
            dtype=float,
        )
        weights = np.asarray(
            [
                hf.inverse_rmse_weight(row.get("sarima_inner_rmse")),
                hf.inverse_rmse_weight(row.get("svr_inner_rmse")),
                hf.inverse_rmse_weight(row.get("lightgbm_inner_rmse")),
            ],
            dtype=float,
        )

        valid_weighted = np.isfinite(preds) & (weights > 0)
        valid_any = np.isfinite(preds)

        if valid_weighted.any():
            weighted_preds.append(float(np.average(preds[valid_weighted], weights=weights[valid_weighted])))
        elif valid_any.any():
            weighted_preds.append(float(np.nanmean(preds[valid_any])))
        else:
            weighted_preds.append(np.nan)

    merged["weighted_ensemble_pred"] = weighted_preds
    return merged


def melt_metrics(metrics_df, prefix):
    if metrics_df.empty:
        return pd.DataFrame(columns=["series_id", "region", "commodity_name", "model", f"{prefix}_rmse", f"{prefix}_r2", f"{prefix}_rows"])

    renamed = metrics_df.copy()
    renamed[f"{prefix}_rmse"] = renamed["rmse"]
    renamed[f"{prefix}_r2"] = renamed["r2"]
    renamed[f"{prefix}_rows"] = renamed["rows_evaluated"]
    keep_cols = ["series_id", "region", "commodity_name", "model", f"{prefix}_rmse", f"{prefix}_r2", f"{prefix}_rows"]
    return renamed[keep_cols]


def compute_model_metrics(frame):
    if frame.empty:
        empty_series = pd.DataFrame(columns=["series_id", "region", "commodity_name", "model", "rmse", "mae", "r2", "rows_evaluated"])
        empty_global = pd.DataFrame(columns=["model", "rmse", "mae", "r2", "rows_evaluated"])
        return empty_series, empty_global

    prediction_pairs = [(model_name, MODEL_COLUMN_MAP[model_name]) for model_name in APP_MODELS]
    series_metrics = hf.compute_series_metrics(frame, prediction_pairs)
    global_metrics = hf.compute_metrics_table(frame, prediction_pairs)
    return series_metrics, global_metrics


def run_single_sarima(meta_row, series_lookup):
    meta_dict = meta_row.to_dict()
    series_frame = series_lookup[meta_dict["series_id"]]
    return hf.run_sarima_for_series(meta_dict, series_frame)


def run_experiment(run_id, eligible_panel, sarima_ready_manifest, non_linear_ready_manifest, svr_features, lightgbm_features, lightgbm_exog_features):
    series_lookup = {
        series_id: part.copy()
        for series_id, part in eligible_panel.groupby("series_id")
    }

    if sarima_ready_manifest.empty:
        sarima_outputs = []
    else:
        sarima_outputs = Parallel(
            n_jobs=N_JOBS,
            backend=BACKEND,
            batch_size=1,
            verbose=0,
        )(
            delayed(run_single_sarima)(meta_row, series_lookup)
            for _, meta_row in sarima_ready_manifest.iterrows()
        )

    sarima_results_rows = []
    sarima_settings_rows = []
    for result in sarima_outputs:
        if result.get("settings") is not None:
            sarima_settings_rows.append(result["settings"])
        sarima_results_rows.extend(result.get("predictions", []))

    sarima_predictions = pd.DataFrame(sarima_results_rows)
    sarima_model_settings = pd.DataFrame(sarima_settings_rows)

    svr_predictions, svr_model_settings = hf.run_svr_models(
        non_linear_ready_manifest,
        eligible_panel,
        svr_features,
    )

    non_linear_predictions = sarima_predictions.merge(
        svr_predictions[["series_id", "region", "commodity_name", "month", "svr_pred"]],
        on=["series_id", "region", "commodity_name", "month"],
        how="inner",
    )

    lightgbm_predictions, lightgbm_model_settings = hf.run_lightgbm_models(
        non_linear_ready_manifest,
        eligible_panel,
        lightgbm_features,
        lightgbm_exog_features,
    )

    if lightgbm_predictions.empty:
        non_linear_predictions["lightgbm_pred"] = non_linear_predictions["naive_pred"]
    else:
        non_linear_predictions = non_linear_predictions.merge(
            lightgbm_predictions[["series_id", "region", "commodity_name", "month", "lightgbm_pred"]],
            on=["series_id", "region", "commodity_name", "month"],
            how="inner",
        )

    ensemble_weights = (
        sarima_model_settings[["series_id", "sarima_inner_rmse"]]
        .merge(svr_model_settings[["series_id", "svr_inner_rmse"]], on="series_id", how="outer")
        .merge(lightgbm_model_settings[["series_id", "lightgbm_inner_rmse"]], on="series_id", how="outer")
    )

    ensemble_predictions = non_linear_predictions.merge(ensemble_weights, on="series_id", how="left").copy()
    weighted_preds = []
    for _, row in ensemble_predictions.iterrows():
        preds = np.asarray([row["sarima_pred"], row["svr_pred"], row["lightgbm_pred"]], dtype=float)
        weights = np.asarray(
            [
                hf.inverse_rmse_weight(row.get("sarima_inner_rmse")),
                hf.inverse_rmse_weight(row.get("svr_inner_rmse")),
                hf.inverse_rmse_weight(row.get("lightgbm_inner_rmse")),
            ],
            dtype=float,
        )
        valid_weighted = np.isfinite(preds) & (weights > 0)
        valid_any = np.isfinite(preds)
        if valid_weighted.any():
            weighted_preds.append(float(np.average(preds[valid_weighted], weights=weights[valid_weighted])))
        elif valid_any.any():
            weighted_preds.append(float(np.nanmean(preds[valid_any])))
        else:
            weighted_preds.append(np.nan)
    ensemble_predictions["weighted_ensemble_pred"] = weighted_preds

    holdout_series_metrics, holdout_global_metrics = compute_model_metrics(ensemble_predictions)
    holdout_series_metrics = melt_metrics(holdout_series_metrics, prefix="holdout")
    holdout_global_metrics = holdout_global_metrics.rename(columns={"rmse": "holdout_rmse", "r2": "holdout_r2", "rows_evaluated": "holdout_rows"})

    sarima_train_df = compute_sarima_train_predictions(eligible_panel, sarima_model_settings)
    svr_train_df = compute_svr_train_predictions(eligible_panel, svr_model_settings)
    lightgbm_train_df = compute_lightgbm_train_predictions(eligible_panel, lightgbm_model_settings)
    ensemble_train_df = combine_component_predictions(
        sarima_train_df,
        svr_train_df,
        lightgbm_train_df,
        ensemble_weights,
    )

    train_prediction_frame = None
    train_frames = [
        sarima_train_df,
        svr_train_df,
        lightgbm_train_df,
        ensemble_train_df[["series_id", "month", "actual", "weighted_ensemble_pred"]] if not ensemble_train_df.empty else pd.DataFrame(),
    ]
    for frame in train_frames:
        if frame.empty:
            continue
        if train_prediction_frame is None:
            train_prediction_frame = frame.copy()
        else:
            train_prediction_frame = train_prediction_frame.merge(frame, on=["series_id", "month", "actual"], how="outer")

    if train_prediction_frame is None or train_prediction_frame.empty:
        train_series_metrics = pd.DataFrame(columns=["series_id", "region", "commodity_name", "model", "train_rmse", "train_r2", "train_rows"])
        train_global_metrics = pd.DataFrame(columns=["model", "train_rmse", "train_r2", "train_rows"])
    else:
        series_meta = eligible_panel[["series_id", "region", "commodity_name"]].drop_duplicates()
        train_prediction_frame = train_prediction_frame.merge(series_meta, on="series_id", how="left")
        train_prediction_frame = train_prediction_frame[["series_id", "region", "commodity_name", "month", "actual"] + list(MODEL_COLUMN_MAP.values())].copy()
        train_series_raw, train_global_raw = compute_model_metrics(train_prediction_frame)
        train_series_metrics = melt_metrics(train_series_raw, prefix="train")
        train_global_metrics = train_global_raw.rename(columns={"rmse": "train_rmse", "r2": "train_r2", "rows_evaluated": "train_rows"})

    consistency = train_series_metrics.merge(
        holdout_series_metrics,
        on=["series_id", "region", "commodity_name", "model"],
        how="outer",
    )
    consistency["run_id"] = run_id
    consistency["rmse_gap"] = consistency["holdout_rmse"] - consistency["train_rmse"]
    consistency["rmse_ratio"] = np.where(
        consistency["train_rmse"] > 0,
        consistency["holdout_rmse"] / consistency["train_rmse"],
        np.nan,
    )

    global_consistency = train_global_metrics.merge(
        holdout_global_metrics,
        on=["model"],
        how="outer",
    )
    global_consistency["run_id"] = run_id
    global_consistency["rmse_gap"] = global_consistency["holdout_rmse"] - global_consistency["train_rmse"]
    global_consistency["rmse_ratio"] = np.where(
        global_consistency["train_rmse"] > 0,
        global_consistency["holdout_rmse"] / global_consistency["train_rmse"],
        np.nan,
    )

    settings_rows = []
    for _, row in sarima_model_settings.iterrows():
        settings_rows.append({
            "run_id": run_id,
            "series_id": row["series_id"],
            "region": row["region"],
            "commodity_name": row["commodity_name"],
            "model": "SARIMA",
            "inner_rmse": row.get("sarima_inner_rmse"),
            "holdout_rows": row.get("holdout_rows"),
        })
    for _, row in svr_model_settings.iterrows():
        settings_rows.append({
            "run_id": run_id,
            "series_id": row["series_id"],
            "region": row["region"],
            "commodity_name": row["commodity_name"],
            "model": "SVR",
            "inner_rmse": row.get("svr_inner_rmse"),
            "holdout_rows": row.get("holdout_rows"),
        })
    for _, row in lightgbm_model_settings.iterrows():
        settings_rows.append({
            "run_id": run_id,
            "series_id": row["series_id"],
            "region": row["region"],
            "commodity_name": row["commodity_name"],
            "model": "LightGBM",
            "inner_rmse": row.get("lightgbm_inner_rmse"),
            "holdout_rows": row.get("holdout_rows"),
        })

    return {
        "series_consistency": consistency.sort_values(["model", "commodity_name"]).reset_index(drop=True),
        "global_consistency": global_consistency.sort_values(["model"]).reset_index(drop=True),
        "settings": pd.DataFrame(settings_rows).sort_values(["model", "commodity_name"]).reset_index(drop=True),
    }


def main():
    eligible_panel, eligible_series_manifest, regional_exogenous_feature_columns = web_export.prepare_base_panel()
    focus_manifest = select_focus_manifest(eligible_series_manifest)
    focus_series_ids = set(focus_manifest["series_id"])
    eligible_panel = eligible_panel.loc[eligible_panel["series_id"].isin(focus_series_ids)].copy()
    eligible_series_manifest = focus_manifest.copy()

    sarima_readiness = build_sarima_readiness(eligible_series_manifest, eligible_panel)
    sarima_ready_manifest = sarima_readiness.loc[
        sarima_readiness["walk_forward_ready"].fillna(False)
    ].copy()
    non_linear_ready_manifest = sarima_ready_manifest.copy()

    svr_features = [feature for feature in SVR_BASE_FEATURES if feature in eligible_panel.columns]
    lightgbm_base_features = [feature for feature in LIGHTGBM_BASE_FEATURES if feature in eligible_panel.columns]
    lightgbm_exog_candidates = [feature for feature in regional_exogenous_feature_columns if feature in eligible_panel.columns]
    if lightgbm_exog_candidates:
        lightgbm_exog_coverage = eligible_panel[lightgbm_exog_candidates].notna().mean().sort_values(ascending=False)
        lightgbm_exog_features = lightgbm_exog_coverage.loc[lightgbm_exog_coverage >= 0.60].index.tolist()[:8]
    else:
        lightgbm_exog_features = []
    lightgbm_features = lightgbm_base_features + lightgbm_exog_features

    selection_report = eligible_series_manifest[["series_id", "region", "commodity_name", "months_total", "yoy_rows"]].copy()
    selection_report["selection_note"] = "Region VI focus set; ordered by dashboard R2 preference with fallback fill."
    selection_report.to_csv(OUTPUT_DIR / "selected_series.csv", index=False)

    run_series_frames = []
    run_global_frames = []
    run_settings_frames = []

    print(f"Running {RUN_COUNT} repeated experiments for {len(eligible_series_manifest)} {TARGET_REGION} series.", flush=True)
    for run_id in range(1, RUN_COUNT + 1):
        start_time = time.perf_counter()
        print(f"Run {run_id}/{RUN_COUNT} started.", flush=True)
        run_output = run_experiment(
            run_id,
            eligible_panel,
            sarima_ready_manifest,
            non_linear_ready_manifest,
            svr_features,
            lightgbm_features,
            lightgbm_exog_features,
        )
        elapsed = time.perf_counter() - start_time
        print(f"Run {run_id}/{RUN_COUNT} finished in {elapsed:.1f}s.", flush=True)
        run_series_frames.append(run_output["series_consistency"])
        run_global_frames.append(run_output["global_consistency"])
        run_settings_frames.append(run_output["settings"])

    run_series_metrics = pd.concat(run_series_frames, ignore_index=True) if run_series_frames else pd.DataFrame()
    run_global_metrics = pd.concat(run_global_frames, ignore_index=True) if run_global_frames else pd.DataFrame()
    run_settings = pd.concat(run_settings_frames, ignore_index=True) if run_settings_frames else pd.DataFrame()

    if run_series_metrics.empty:
        raise ValueError("No run metrics were produced.")

    summary_series = (
        run_series_metrics.groupby(["series_id", "region", "commodity_name", "model"], as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            train_rmse_mean=("train_rmse", "mean"),
            train_rmse_std=("train_rmse", "std"),
            holdout_rmse_mean=("holdout_rmse", "mean"),
            holdout_rmse_std=("holdout_rmse", "std"),
            rmse_gap_mean=("rmse_gap", "mean"),
            rmse_gap_std=("rmse_gap", "std"),
            rmse_ratio_mean=("rmse_ratio", "mean"),
            holdout_r2_mean=("holdout_r2", "mean"),
            holdout_r2_std=("holdout_r2", "std"),
        )
        .sort_values(["model", "commodity_name"])
        .reset_index(drop=True)
    )

    summary_global = (
        run_global_metrics.groupby(["model"], as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            train_rmse_mean=("train_rmse", "mean"),
            train_rmse_std=("train_rmse", "std"),
            holdout_rmse_mean=("holdout_rmse", "mean"),
            holdout_rmse_std=("holdout_rmse", "std"),
            rmse_gap_mean=("rmse_gap", "mean"),
            rmse_gap_std=("rmse_gap", "std"),
            rmse_ratio_mean=("rmse_ratio", "mean"),
            holdout_r2_mean=("holdout_r2", "mean"),
            holdout_r2_std=("holdout_r2", "std"),
        )
        .sort_values(["model"])
        .reset_index(drop=True)
    )

    run_series_metrics.to_csv(OUTPUT_DIR / "run_series_consistency.csv", index=False)
    run_global_metrics.to_csv(OUTPUT_DIR / "run_global_consistency.csv", index=False)
    run_settings.to_csv(OUTPUT_DIR / "run_model_settings.csv", index=False)
    summary_series.to_csv(OUTPUT_DIR / "summary_series_consistency.csv", index=False)
    summary_global.to_csv(OUTPUT_DIR / "summary_global_consistency.csv", index=False)

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "target_region": TARGET_REGION,
        "target_label": TARGET_LABEL,
        "run_count": RUN_COUNT,
        "series_selected": selection_report["commodity_name"].tolist(),
        "output_files": [
            "selected_series.csv",
            "run_series_consistency.csv",
            "run_global_consistency.csv",
            "run_model_settings.csv",
            "summary_series_consistency.csv",
            "summary_global_consistency.csv",
        ],
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Region VI consistency outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
