import constants as const

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

import lightgbm as lgb
import optuna

WINDOW_MIN_MONTHS = const.WINDOW_MIN_MONTHS
WINDOW_MAX_MONTHS = const.WINDOW_MAX_MONTHS

TARGET_COL = const.TARGET_COL
TARGET_LABEL = const.TARGET_LABEL
MIN_TRAIN_TARGET_ROWS = const.MIN_TRAIN_TARGET_ROWS
FORECAST_HORIZON = const.FORECAST_HORIZON
ROLLING_STEP = const.ROLLING_STEP
SEASONAL_PERIOD = const.SEASONAL_PERIOD

HOLDOUT_RATIO = const.HOLDOUT_RATIO
MIN_TRAIN_OBS_SARIMA = const.MIN_TRAIN_OBS_SARIMA
INNER_FOLDS_SARIMA = const.INNER_FOLDS_SARIMA
SARIMA_P_VALUES = const.SARIMA_P_VALUES
SARIMA_Q_VALUES = const.SARIMA_Q_VALUES
SARIMA_P_SEASONAL_VALUES = const.SARIMA_P_SEASONAL_VALUES
SARIMA_Q_SEASONAL_VALUES = const.SARIMA_Q_SEASONAL_VALUES
SARIMA_TREND_VALUES = const.SARIMA_TREND_VALUES
SARIMA_N_JOBS = const.SARIMA_N_JOBS
SARIMA_MAXITER = const.SARIMA_MAXITER

SVR_PARAM_GRID = const.SVR_PARAM_GRID
MIN_TRAIN_OBS_ML = const.MIN_TRAIN_OBS_SARIMA
INNER_FOLDS_ML = const.INNER_FOLDS_SARIMA

LIGHTGBM_BASE_FEATURES = const.LIGHTGBM_BASE_FEATURES
LIGHTGBM_TRIALS = const.LIGHTGBM_TRIALS
LIGHTGBM_RANDOM_STATE = const.LIGHTGBM_RANDOM_STATE

def transform_series(series, mode):
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if mode == "mom_pct":
        transformed = series.pct_change(fill_method=None) * 100
    elif mode == "diff":
        transformed = series.diff()
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")
    return transformed.replace([np.inf, -np.inf], np.nan)

def safe_adf_pvalue(series, autolag="AIC"):
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 12 or clean.nunique() < 3:
        return np.nan
    try:
        return float(adfuller(clean, autolag=autolag)[1])
    except Exception:
        return np.nan

def run_granger_direction(target, driver, max_lag=6, min_obs=36, alpha=0.05):
    paired = (
        pd.concat([target.rename("target"), driver.rename("driver")], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    result = {
        "overlap_points": len(paired),
        "start_month": paired.index.min().date().isoformat() if not paired.empty else "",
        "end_month": paired.index.max().date().isoformat() if not paired.empty else "",
        **{f"lag_{i}_p": np.nan for i in range(1, max_lag + 1)},
    }

    if len(paired) < max(max_lag * 3, min_obs):
        return {
            **result,
            "best_lag": np.nan,
            "min_p_value": np.nan,
            "significant": False,
            "status": "insufficient_observations",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tests = grangercausalitytests(paired[["target", "driver"]], maxlag=max_lag, verbose=False)

        pvalues = {i: float(tests[i][0]["ssr_ftest"][1]) for i in range(1, max_lag + 1)}
        best_lag = min(pvalues, key=pvalues.get)
        min_p = pvalues[best_lag]

        return {
            **result,
            **{f"lag_{i}_p": round(p, 6) for i, p in pvalues.items()},
            "best_lag": best_lag,
            "min_p_value": round(min_p, 6),
            "significant": min_p < alpha,
            "status": "ok",
        }

    except Exception as exc:
        return {
            **result,
            "best_lag": np.nan,
            "min_p_value": np.nan,
            "significant": False,
            "status": f"error: {type(exc).__name__}",
        }

def benjamini_hochberg(pvalues):
    values = pd.to_numeric(pvalues, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)

    valid = values.dropna().sort_values()
    if valid.empty:
        return adjusted

    n = len(valid)
    ranks = np.arange(1, n + 1, dtype=float)

    raw = (valid.values * n) / ranks
    monotonic = np.minimum.accumulate(raw[::-1])[::-1]
    monotonic = np.clip(monotonic, 0, 1)

    adjusted.loc[valid.index] = monotonic
    return adjusted

def apply_linear_interpolation(group, max_gap=2):
    group = group.copy()
    group["month"] = pd.to_datetime(group["month"], errors="coerce")
    group["price"] = pd.to_numeric(group["price"], errors="coerce")

    group = (
        group.sort_values("month")
        .groupby("month", as_index=False)
        .agg(price=("price", "median"))
    )

    full_months = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")

    group = (
        group.set_index("month")
        .reindex(full_months)
        .rename_axis("month")
        .reset_index()
    )
    is_missing = group["price"].isna()
    run_id = is_missing.ne(is_missing.shift()).cumsum()
    missing_run_sizes = group[is_missing].groupby(run_id[is_missing]).size()

    if not missing_run_sizes.empty and missing_run_sizes.max() > max_gap:
        return None

    group["price"] = group["price"].interpolate(method="linear", limit=max_gap, limit_direction="both")
    return group

def add_basic_price_features(panel):
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")

    panel["series_id"] = (
        panel["region"].astype(str) + "|" +
        panel["commodity_name"].astype(str)
    )

    panel = panel.sort_values(["series_id", "month"]).reset_index(drop=True)

    panel["price_target"] = panel["price"]
    panel["price_diff"] = panel.groupby("series_id")["price"].diff()
    panel["mom_inflation"] = panel.groupby("series_id")["price"].pct_change(fill_method=None) * 100
    panel["yoy_inflation"] = panel.groupby("series_id")["price"].pct_change(12, fill_method=None) * 100
    panel["log_price"] = np.where(panel["price"] > 0, np.log(panel["price"]), np.nan)
    panel["log_price_diff"] = panel.groupby("series_id")["log_price"].diff()
    panel["log_price_yoy"] = panel.groupby("series_id")["log_price"].diff(12)

    panel["base_price"] = panel.groupby("series_id")["price"].transform("first")
    panel["price_index"] = panel["price"] / panel["base_price"]

    for lag in [1, 2, 3, 6, 12]:
        panel[f"price_lag_{lag}"] = panel.groupby("series_id")["price"].shift(lag)
        panel[f"price_diff_lag_{lag}"] = panel.groupby("series_id")["price_diff"].shift(lag)
        panel[f"mom_lag_{lag}"] = panel.groupby("series_id")["mom_inflation"].shift(lag)
        panel[f"yoy_lag_{lag}"] = panel.groupby("series_id")["yoy_inflation"].shift(lag)
        panel[f"price_index_lag_{lag}"] = panel.groupby("series_id")["price_index"].shift(lag)
        panel[f"log_price_lag_{lag}"] = panel.groupby("series_id")["log_price"].shift(lag)
        panel[f"log_price_yoy_lag_{lag}"] = panel.groupby("series_id")["log_price_yoy"].shift(lag)

    for window in [3, 6, 12]:
        lagged_yoy = panel.groupby("series_id")["yoy_inflation"].shift(1)
        lagged_price_index = panel.groupby("series_id")["price_index"].shift(1)

        panel[f"yoy_roll_mean_{window}"] = (
            lagged_yoy.groupby(panel["series_id"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        panel[f"yoy_roll_std_{window}"] = (
            lagged_yoy.groupby(panel["series_id"])
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )
        panel[f"price_index_roll_mean_{window}"] = (
            lagged_price_index.groupby(panel["series_id"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        panel[f"price_index_roll_std_{window}"] = (
            lagged_price_index.groupby(panel["series_id"])
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )

    panel["yoy_acceleration"] = panel.groupby("series_id")["yoy_inflation"].diff()
    panel["price_index_gap_from_roll3"] = panel["price_index_lag_1"] - panel["price_index_roll_mean_3"]
    panel["price_index_gap_from_roll6"] = panel["price_index_lag_1"] - panel["price_index_roll_mean_6"]
    panel["price_index_momentum_3"] = panel["price_index_lag_1"] - panel["price_index_lag_3"]

    panel["month_num"] = panel["month"].dt.month
    panel["quarter_num"] = panel["month"].dt.quarter
    panel["month_sin"] = np.sin(2 * np.pi * panel["month_num"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month_num"] / 12)
    panel["quarter_sin"] = np.sin(2 * np.pi * panel["quarter_num"] / 4)
    panel["quarter_cos"] = np.cos(2 * np.pi * panel["quarter_num"] / 4)

    panel["yoy_lag1_x_roll3"] = panel["yoy_lag_1"] * panel["yoy_roll_mean_3"]
    panel["yoy_lag1_x_roll6"] = panel["yoy_lag_1"] * panel["yoy_roll_mean_6"]
    panel["yoy_lag1_x_month_sin"] = panel["yoy_lag_1"] * panel["month_sin"]
    panel["yoy_lag1_x_month_cos"] = panel["yoy_lag_1"] * panel["month_cos"]

    panel["post_2008_break"] = (panel["month"] >= pd.Timestamp("2008-01-01")).astype(int)
    panel["post_2020_break"] = (panel["month"] >= pd.Timestamp("2020-03-01")).astype(int)
    panel["post_2024_break"] = (panel["month"] >= pd.Timestamp("2024-01-01")).astype(int)
    panel["covid_shock_window"] = (
        (panel["month"] >= pd.Timestamp("2020-03-01")) &
        (panel["month"] <= pd.Timestamp("2021-12-01"))
    ).astype(int)

    return panel


def add_exogenous_lags(panel, drivers, lags=[1, 2, 3]):
    panel = panel.copy()
    exog_cols = []

    for driver in drivers:
        if driver not in panel.columns:
            continue

        for lag in lags:
            col_name = f"{driver}_lag_{lag}"
            panel[col_name] = panel.groupby("series_id")[driver].shift(lag)
            exog_cols.append(col_name)

    return panel, exog_cols

def build_series_manifest(panel):
    windowed_parts = []
    manifest_rows = []

    for series_id, part in panel.groupby("series_id"):
        ordered = part.sort_values("month").reset_index(drop=True)

        if len(ordered) < WINDOW_MIN_MONTHS:
            continue

        window = ordered.tail(WINDOW_MAX_MONTHS).copy()
        windowed_parts.append(window)

        manifest_rows.append({
            "series_id": series_id,
            "region": window["region"].iloc[0],
            "commodity_name": window["commodity_name"].iloc[0],
            "start_month": window["month"].min(),
            "end_month": window["month"].max(),
            "months_total": len(window),
            "price_rows": int(window["price"].notna().sum()),
            "mom_rows": int(window["mom_inflation"].notna().sum()),
            "yoy_rows": int(window["yoy_inflation"].notna().sum()),
            "eligible": True,
        })

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["months_total", "yoy_rows", "region", "commodity_name"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    window_panel = pd.concat(windowed_parts, ignore_index=True) if windowed_parts else panel.iloc[0:0].copy()
    return manifest, window_panel

def lagged_autocorr(series, lag):
    series = pd.Series(series).dropna()
    if len(series) <= lag:
        return np.nan
    return series.autocorr(lag=lag)

# FOR SARIMA
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return np.nan
    sse = float(np.sum(np.square(y_true - y_pred)))
    sst = float(np.sum(np.square(y_true - y_true.mean())))
    if sst == 0:
        return np.nan
    return 1 - (sse / sst)


def expanding_splits(length, folds, min_train):
    splits = []
    if length < min_train + folds:
        return splits
    for pos in range(length - folds, length):
        if pos >= min_train:
            splits.append((np.arange(0, pos), np.array([pos])))
    return splits


def seasonal_naive_forecast(train_df, forecast_month, target_col):
    history = train_df.loc[train_df[target_col].notna(), ["month", target_col]].copy()
    if history.empty:
        return np.nan

    same_month = history.loc[history["month"].dt.month == pd.Timestamp(forecast_month).month]
    if not same_month.empty:
        return float(same_month.iloc[-1][target_col])

    return float(history.iloc[-1][target_col])


def compute_metrics_table(frame, prediction_pairs):
    rows = []
    for model_name, column in prediction_pairs:
        valid = frame.loc[frame["actual"].notna() & frame[column].notna()].copy()
        if valid.empty:
            rows.append({
                "model": model_name,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "rows_evaluated": 0
            })
            continue

        rows.append({
            "model": model_name,
            "rmse": rmse(valid["actual"], valid[column]),
            "mae": mae(valid["actual"], valid[column]),
            "r2": r2_from_arrays(valid["actual"], valid[column]),
            "rows_evaluated": int(len(valid))
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def compute_series_metrics(frame, prediction_pairs):
    rows = []
    for series_id, part in frame.groupby("series_id"):
        for model_name, column in prediction_pairs:
            valid = part.loc[part["actual"].notna() & part[column].notna()].copy()
            if valid.empty:
                continue

            rows.append({
                "series_id": series_id,
                "region": valid["region"].iloc[0],
                "commodity_name": valid["commodity_name"].iloc[0],
                "model": model_name,
                "rmse": rmse(valid["actual"], valid[column]),
                "mae": mae(valid["actual"], valid[column]),
                "r2": r2_from_arrays(valid["actual"], valid[column]),
                "rows_evaluated": int(len(valid))
            })

    return pd.DataFrame(rows)


def compute_diagnostics(frame, prediction_pairs):
    rows = []
    for model_name, column in prediction_pairs:
        valid = frame.loc[frame["actual"].notna() & frame[column].notna()].copy()

        if valid.empty:
            rows.append({
                "model": model_name,
                "residual_mean": np.nan,
                "residual_std": np.nan,
                "residual_lag1_autocorr": np.nan,
                "ljungbox_pvalue_lag1": np.nan,
                "ljungbox_pvalue_lag12": np.nan,
            })
            continue

        residuals = valid["actual"] - valid[column]

        try:
            lb_1 = acorr_ljungbox(residuals, lags=[1], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_1 = np.nan

        try:
            lb_12 = acorr_ljungbox(residuals, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_12 = np.nan

        rows.append({
            "model": model_name,
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std(ddof=1)),
            "residual_lag1_autocorr": float(residuals.autocorr(lag=1)) if len(residuals) > 1 else np.nan,
            "ljungbox_pvalue_lag1": lb_1,
            "ljungbox_pvalue_lag12": lb_12,
        })

    return pd.DataFrame(rows)


def run_sarima_for_series(meta_dict, df):
    meta = dict(meta_dict)
    df = df.sort_values("month").reset_index(drop=True).copy()

    valid_positions = df.index[df[TARGET_COL].notna()].tolist()
    if len(valid_positions) < MIN_TRAIN_OBS_SARIMA + 1:
        return {"settings": None, "predictions": [], "series_id": meta.get("series_id")}

    holdout_count = max(1, int(np.ceil(len(valid_positions) * HOLDOUT_RATIO)))
    holdout_count = min(holdout_count, len(valid_positions) - MIN_TRAIN_OBS_SARIMA)
    if holdout_count <= 0:
        return {"settings": None, "predictions": [], "series_id": meta.get("series_id")}

    test_positions = valid_positions[-holdout_count:]
    first_test_position = test_positions[0]

    first_train = df.iloc[:first_test_position].copy()
    tuning = first_train.loc[first_train[TARGET_COL].notna(), ["month", TARGET_COL]].copy().reset_index(drop=True)
    if len(tuning) < MIN_TRAIN_OBS_SARIMA:
        return {"settings": None, "predictions": [], "series_id": meta.get("series_id")}

    inner_splits = expanding_splits(
        length=len(tuning),
        folds=INNER_FOLDS_SARIMA,
        min_train=max(MIN_TRAIN_OBS_SARIMA, min(36, len(tuning) - 1))
    )
    if not inner_splits:
        return {"settings": None, "predictions": [], "series_id": meta.get("series_id")}

    base_d = int(meta["recommended_d"]) if pd.notna(meta.get("recommended_d")) else 0
    base_D = int(meta["recommended_D"]) if pd.notna(meta.get("recommended_D")) else 0

    d_candidates = sorted(set([max(0, base_d - 1), base_d, min(2, base_d + 1)]))
    D_candidates = sorted(set([max(0, base_D - 1), base_D, min(1, base_D + 1)]))

    naive_cv_preds = []
    seasonal_cv_preds = []
    cv_actuals = []

    for train_idx, val_idx in inner_splits:
        fit_block = tuning.iloc[train_idx].copy()
        val_block = tuning.iloc[val_idx].copy()

        naive_cv_preds.append(float(fit_block[TARGET_COL].iloc[-1]))

        seasonal_cv = seasonal_naive_forecast(fit_block, val_block["month"].iloc[0], TARGET_COL)
        if pd.isna(seasonal_cv):
            seasonal_cv = float(fit_block[TARGET_COL].iloc[-1])
        seasonal_cv_preds.append(float(seasonal_cv))

        cv_actuals.append(float(val_block[TARGET_COL].iloc[0]))

    baseline_rmse = rmse(cv_actuals, naive_cv_preds)
    seasonal_baseline_rmse = rmse(cv_actuals, seasonal_cv_preds)

    best_sarima_aic = float("inf")
    best_sarima_rmse = float("inf")
    best_sarima_bias = float("inf")
    best_sarima_order = (1, base_d, 1)
    best_sarima_seasonal = (0, base_D, 0, SEASONAL_PERIOD)
    best_sarima_trend = "n"

    for d_value in d_candidates:
        for D_value in D_candidates:
            for trend_value in SARIMA_TREND_VALUES:
                if d_value > 0 or D_value > 0:
                    if trend_value == "c":
                        continue

                for p in SARIMA_P_VALUES:
                    for q in SARIMA_Q_VALUES:
                        for P in SARIMA_P_SEASONAL_VALUES:
                            for Q in SARIMA_Q_SEASONAL_VALUES:
                                order = (p, d_value, q)
                                seasonal_order = (P, D_value, Q, SEASONAL_PERIOD)

                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        fitted = SARIMAX(
                                            tuning[TARGET_COL],
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            trend=trend_value,
                                            enforce_stationarity=True,
                                            enforce_invertibility=True,
                                        ).fit(disp=False, maxiter=SARIMA_MAXITER)

                                    current_aic = float(fitted.aic) if pd.notna(fitted.aic) else float("inf")
                                except Exception:
                                    continue

                                cv_preds = []
                                cv_targets = []

                                for train_idx, val_idx in inner_splits:
                                    y_fit = tuning[TARGET_COL].iloc[train_idx]
                                    y_val = tuning[TARGET_COL].iloc[val_idx]

                                    try:
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore")
                                            cv_fit = SARIMAX(
                                                y_fit,
                                                order=order,
                                                seasonal_order=seasonal_order,
                                                trend=trend_value,
                                                enforce_stationarity=True,
                                                enforce_invertibility=True,
                                            ).fit(disp=False, maxiter=300)

                                        cv_pred = float(np.asarray(cv_fit.forecast(steps=1))[0])

                                        if not np.isfinite(cv_pred) or abs(cv_pred) > 1e6:
                                            cv_pred = float(y_fit.iloc[-1])

                                        cv_preds.append(cv_pred)
                                    except Exception:
                                        cv_preds.append(float(y_fit.iloc[-1]))

                                    cv_targets.append(float(y_val.iloc[0]))

                                current_rmse = rmse(cv_targets, cv_preds)
                                current_bias = float(np.mean(np.asarray(cv_targets) - np.asarray(cv_preds)))

                                if current_rmse < best_sarima_rmse or (
                                    np.isclose(current_rmse, best_sarima_rmse, equal_nan=False)
                                    and current_aic < best_sarima_aic
                                ):
                                    best_sarima_aic = current_aic
                                    best_sarima_rmse = current_rmse
                                    best_sarima_bias = current_bias
                                    best_sarima_order = order
                                    best_sarima_seasonal = seasonal_order
                                    best_sarima_trend = trend_value

    settings_row = {
        "series_id": meta["series_id"],
        "region": meta["region"],
        "commodity_name": meta["commodity_name"],
        "recommended_d": base_d,
        "recommended_D": base_D,
        "tested_d_values": str(d_candidates),
        "tested_D_values": str(D_candidates),
        "sarima_order": str(best_sarima_order),
        "sarima_seasonal_order": str(best_sarima_seasonal),
        "sarima_trend": best_sarima_trend,
        "sarima_best_aic": best_sarima_aic,
        "sarima_inner_rmse": best_sarima_rmse,
        "sarima_inner_bias": best_sarima_bias,
        "naive_inner_rmse": baseline_rmse,
        "seasonal_naive_inner_rmse": seasonal_baseline_rmse,
        "holdout_rows": len(test_positions),
    }

    prediction_rows = []

    for holdout_position in test_positions:
        train_block = df.iloc[:holdout_position].copy()
        row = df.iloc[holdout_position].copy()
        actual = row[TARGET_COL]

        if pd.isna(actual):
            continue

        tuning_history = train_block.loc[train_block[TARGET_COL].notna(), ["month", TARGET_COL]].copy().reset_index(drop=True)
        if tuning_history.empty:
            continue

        naive_pred = float(tuning_history[TARGET_COL].iloc[-1])

        seasonal_pred = seasonal_naive_forecast(tuning_history, row["month"], TARGET_COL)
        if pd.isna(seasonal_pred):
            seasonal_pred = naive_pred

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sarima_fit = SARIMAX(
                    tuning_history[TARGET_COL],
                    order=best_sarima_order,
                    seasonal_order=best_sarima_seasonal,
                    trend=best_sarima_trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=300)

            sarima_pred = float(np.asarray(sarima_fit.forecast(steps=1))[0])
            if not np.isfinite(sarima_pred) or abs(sarima_pred) > 1e6:
                sarima_pred = naive_pred
        except Exception:
            sarima_pred = naive_pred

        prediction_rows.append({
            "series_id": meta["series_id"],
            "region": meta["region"],
            "commodity_name": meta["commodity_name"],
            "month": row["month"],
            "actual": float(actual),
            "naive_pred": float(naive_pred),
            "seasonal_naive_pred": float(seasonal_pred),
            "sarima_pred": float(sarima_pred),
        })

    return {"settings": settings_row, "predictions": prediction_rows, "series_id": meta.get("series_id")}

def build_ml_tuning_frame(df, feature_cols, exog_cols=None):
    exog_cols = exog_cols or []
    df = df.sort_values("month").reset_index(drop=True).copy()

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

    for col in exog_cols:
        first_train[col] = first_train[col].ffill().bfill()

    tuning = (
        first_train.loc[
            first_train[feature_cols + [TARGET_COL]].notna().all(axis=1),
            ["month"] + feature_cols + [TARGET_COL],
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(tuning) < max(18, min(40, len(feature_cols) * 2)):
        return None

    inner_splits = expanding_splits(
        length=len(tuning),
        folds=INNER_FOLDS_ML,
        min_train=max(18, min(24, len(tuning) - 1)),
    )
    if not inner_splits:
        return None

    return {
        "test_positions": test_positions,
        "tuning": tuning,
        "inner_splits": inner_splits,
    }


def run_svr_models(manifest, panel, feature_cols):
    prediction_rows = []
    svr_settings_rows = []

    for _, meta in manifest.iterrows():
        df = panel.loc[panel["series_id"] == meta["series_id"]].sort_values("month").reset_index(drop=True)
        prepared = build_ml_tuning_frame(df, feature_cols)

        if prepared is None:
            continue

        test_positions = prepared["test_positions"]
        tuning = prepared["tuning"]

        if len(tuning) < MIN_TRAIN_OBS_ML:
            continue

        n_splits = min(INNER_FOLDS_ML, max(2, len(tuning) - MIN_TRAIN_OBS_ML))
        if n_splits < 2:
            continue

        time_cv = TimeSeriesSplit(n_splits=n_splits)

        svr_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("svr", SVR(kernel="rbf"))
        ])

        grid = GridSearchCV(
            estimator=svr_pipeline,
            param_grid=SVR_PARAM_GRID,
            scoring="neg_root_mean_squared_error",
            cv=time_cv,
            n_jobs=1,
            refit=True,
        )

        X_tuning = tuning[feature_cols]
        y_tuning = tuning[TARGET_COL]

        try:
            grid.fit(X_tuning, y_tuning)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_rmse = -float(grid.best_score_)
        except Exception:
            continue

        svr_settings_rows.append({
            "series_id": meta["series_id"],
            "region": meta["region"],
            "commodity_name": meta["commodity_name"],
            "selected_features": str(feature_cols),
            "svr_best_params": str(best_params),
            "svr_inner_rmse": best_rmse,
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
            seasonal_pred = seasonal_naive_forecast(history, row["month"], TARGET_COL)
            if pd.isna(seasonal_pred):
                seasonal_pred = naive_pred

            model_train = train_block.loc[
                train_block[feature_cols + [TARGET_COL]].notna().all(axis=1),
                feature_cols + [TARGET_COL],
            ].copy()

            feature_ready = bool(feature_cols) and row[feature_cols].notna().all()

            if len(model_train) >= MIN_TRAIN_OBS_ML and feature_ready:
                X_train = model_train[feature_cols]
                y_train = model_train[TARGET_COL]
                X_test = pd.DataFrame([row[feature_cols].values], columns=feature_cols)

                try:
                    final_model = Pipeline([
                        ("scaler", RobustScaler()),
                        ("svr", SVR(
                            kernel="rbf",
                            C=best_params["svr__C"],
                            epsilon=best_params["svr__epsilon"],
                            gamma=best_params["svr__gamma"],
                        ))
                    ])
                    final_model.fit(X_train, y_train)
                    svr_pred = float(final_model.predict(X_test)[0])
                except Exception:
                    svr_pred = naive_pred
            else:
                svr_pred = naive_pred

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

    return pd.DataFrame(prediction_rows), pd.DataFrame(svr_settings_rows)

def fill_lightgbm_exog_from_history(train_block, row, exog_cols):
    row = row.copy()
    for col in exog_cols:
        if pd.isna(row.get(col)):
            last_known = train_block[col].dropna()
            if not last_known.empty:
                row[col] = float(last_known.iloc[-1])
    return row

def tune_lightgbm_feature_set(tuning, inner_splits, feature_cols):
    if not feature_cols:
        return None, float("inf")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 180, 520),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 25),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        }

        cv_preds = []
        cv_actuals = []

        for train_idx, val_idx in inner_splits:
            X_fit = tuning[feature_cols].iloc[train_idx]
            X_val = tuning[feature_cols].iloc[val_idx]
            y_fit = tuning[TARGET_COL].iloc[train_idx]
            y_val = tuning[TARGET_COL].iloc[val_idx]

            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=LIGHTGBM_RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,
                force_col_wise=True,
                max_bin=127,
                **params,
            )
            model.fit(X_fit, y_fit)
            cv_preds.extend(model.predict(X_val).tolist())
            cv_actuals.extend(y_val.tolist())

        return rmse(cv_actuals, cv_preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=LIGHTGBM_RANDOM_STATE),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=LIGHTGBM_TRIALS, show_progress_bar=False)

    return study.best_params, float(study.best_value)


def run_lightgbm_models(manifest, panel, feature_cols, exog_cols):
    prediction_rows = []
    settings_rows = []

    for _, meta in manifest.iterrows():
        df = panel.loc[panel["series_id"] == meta["series_id"]].sort_values("month").reset_index(drop=True)
        prepared = build_ml_tuning_frame(df, feature_cols, exog_cols=exog_cols)

        if prepared is None:
            continue

        test_positions = prepared["test_positions"]
        tuning = prepared["tuning"]
        inner_splits = prepared["inner_splits"]

        best_params, best_rmse = tune_lightgbm_feature_set(tuning, inner_splits, feature_cols)
        if best_params is None:
            continue

        feature_probe = lgb.LGBMRegressor(
            objective="regression",
            random_state=LIGHTGBM_RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
            force_col_wise=True,
            max_bin=127,
            **best_params,
        )
        feature_probe.fit(tuning[feature_cols], tuning[TARGET_COL])

        importance_frame = (
            pd.DataFrame({"feature": feature_cols, "importance": feature_probe.feature_importances_})
            .sort_values(["importance", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        top_features = ", ".join(
            importance_frame.loc[importance_frame["importance"] > 0].head(6)["feature"].tolist()
        )

        settings_rows.append({
            "series_id": meta["series_id"],
            "region": meta["region"],
            "commodity_name": meta["commodity_name"],
            "selected_features": str(feature_cols),
            "lightgbm_exogenous_features": str(exog_cols),
            "lightgbm_top_features": top_features,
            "lightgbm_params": str(best_params),
            "lightgbm_inner_rmse": best_rmse,
            "holdout_rows": len(test_positions),
        })

        for holdout_position in test_positions:
            train_block = df.iloc[:holdout_position].copy()
            raw_row = df.iloc[holdout_position].copy()
            actual = raw_row[TARGET_COL]

            if pd.isna(actual):
                continue

            history = train_block.loc[train_block[TARGET_COL].notna(), ["month", TARGET_COL]].copy().reset_index(drop=True)
            if history.empty:
                continue

            naive_pred = float(history[TARGET_COL].iloc[-1])
            seasonal_pred = seasonal_naive_forecast(history, raw_row["month"], TARGET_COL)
            if pd.isna(seasonal_pred):
                seasonal_pred = naive_pred

            predict_row = fill_lightgbm_exog_from_history(train_block, raw_row, exog_cols)

            model_train = train_block.copy()
            for col in exog_cols:
                model_train[col] = model_train[col].ffill().bfill()

            model_train = model_train.loc[
                model_train[feature_cols + [TARGET_COL]].notna().all(axis=1),
                feature_cols + [TARGET_COL],
            ].copy()

            feature_ready = bool(feature_cols) and predict_row[feature_cols].notna().all()

            if len(model_train) >= MIN_TRAIN_OBS_ML and feature_ready:
                test_features = pd.DataFrame([predict_row[feature_cols].values], columns=feature_cols)

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
                    lightgbm_pred = float(model.predict(test_features)[0])
                except Exception:
                    lightgbm_pred = naive_pred
            else:
                lightgbm_pred = naive_pred

            prediction_rows.append({
                "series_id": meta["series_id"],
                "region": meta["region"],
                "commodity_name": meta["commodity_name"],
                "month": raw_row["month"],
                "actual": float(actual),
                "naive_pred": float(naive_pred),
                "seasonal_naive_pred": float(seasonal_pred),
                "lightgbm_pred": float(lightgbm_pred),
            })

    return pd.DataFrame(prediction_rows), pd.DataFrame(settings_rows)

def compute_series_diagnostics(frame, prediction_pairs):
    rows = []

    for (series_id, region, commodity_name), part in frame.groupby(
        ["series_id", "region", "commodity_name"]
    ):
        for model_name, column in prediction_pairs:
            valid = part.loc[
                part["actual"].notna() & part[column].notna()
            ].copy()

            if len(valid) < 12:
                continue

            residuals = valid["actual"] - valid[column]

            try:
                lb_1 = acorr_ljungbox(residuals, lags=[1], return_df=True)["lb_pvalue"].iloc[0]
            except Exception:
                lb_1 = np.nan

            try:
                lb_12 = acorr_ljungbox(residuals, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
            except Exception:
                lb_12 = np.nan

            rows.append({
                "series_id": series_id,
                "region": region,
                "commodity_name": commodity_name,
                "model": model_name,
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std(ddof=1)),
                "residual_lag1_autocorr": float(residuals.autocorr(lag=1)) if len(residuals) > 1 else np.nan,
                "ljungbox_pvalue_lag1": lb_1,
                "ljungbox_pvalue_lag12": lb_12,
                "rows_evaluated": int(len(valid)),
            })

    return pd.DataFrame(rows)
