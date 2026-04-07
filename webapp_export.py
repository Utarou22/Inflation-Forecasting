import ast
from io import StringIO
import json
from html.parser import HTMLParser
from pathlib import Path
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX

import constants as const
import helper_functions as hf


ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "tables"
ARTIFACTS_DIR = ROOT / "artifacts"
WEBAPP_DATA_DIR = ROOT / "webapp" / "data"
WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = const.TARGET_COL
TARGET_LABEL = const.TARGET_LABEL
MIN_TRAIN_OBS_SARIMA = const.MIN_TRAIN_OBS_SARIMA
MIN_TRAIN_OBS_ML = const.MIN_TRAIN_OBS_ML
HOLDOUT_RATIO = const.HOLDOUT_RATIO
SARIMA_MAXITER = const.SARIMA_MAXITER
LIGHTGBM_RANDOM_STATE = const.LIGHTGBM_RANDOM_STATE
MODEL_SERIES_CAP = const.MODEL_SERIES_CAP
MODEL_SERIES_CAP_FALLBACK = const.MODEL_SERIES_CAP_FALLBACK
SVR_BASE_FEATURES = const.SVR_BASE_FEATURES
LIGHTGBM_BASE_FEATURES = const.LIGHTGBM_BASE_FEATURES
FORECAST_MONTHS = 12
APP_MODELS = ["SARIMA", "SVR", "LightGBM", "Weighted Ensemble"]
EXOG_DRIVER_COLUMNS = [
    "diesel_price",
    "food_price_index",
    "max_consecutive_dry_days",
    "days_above_35c",
    "days_rain_above_50mm_extreme",
    "precipitation_mm",
    "avg_mean_temp_c",
]


def parse_literal(value, fallback=None):
    if isinstance(value, (list, tuple, dict)):
        return value
    if pd.isna(value):
        return fallback
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return fallback


class DataFrameHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.current_row = []
        self.current_cell = []
        self.headers = []
        self.rows = []
        self.in_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.in_header = True
        elif tag == "tbody":
            self.in_header = False
        elif tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"th", "td"} and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.in_cell:
            text = "".join(self.current_cell).strip()
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                if self.in_header and not self.headers:
                    self.headers = self.current_row
                elif not self.in_header:
                    self.rows.append(self.current_row)
            self.in_row = False
        elif tag == "thead":
            self.in_header = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def normalize_table(df):
    df = df.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    for column in df.columns:
        if "month" in str(column).lower():
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().any():
                df[column] = converted
                continue

        converted = pd.to_numeric(df[column], errors="coerce")
        non_null_original = df[column].notna().sum()
        non_null_converted = converted.notna().sum()
        if non_null_original > 0 and non_null_converted >= max(1, int(non_null_original * 0.8)):
            df[column] = converted
    return df


def load_html_table(path):
    parser = DataFrameHTMLParser()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    if not parser.headers:
        raise ValueError(f"No table headers found in {path}")
    df = pd.DataFrame(parser.rows, columns=parser.headers)
    return normalize_table(df)


def load_artifact_payloads(model_name, artifact_subdir):
    manifest = pd.read_csv(ARTIFACTS_DIR / "artifact_manifest.csv")
    rows = manifest.loc[
        (manifest["model"] == model_name) & (manifest["status"] == "saved")
    ].copy()

    payloads = {}
    for _, row in rows.iterrows():
        artifact_path = Path(row["artifact_path"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = joblib.load(artifact_path)
        payloads[payload["series_id"]] = payload

    return payloads


def load_ensemble_weight_lookup():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = joblib.load(ARTIFACTS_DIR / "weighted_ensemble" / "ensemble_weights.joblib")

    weights = payload["weights"].copy()
    return {row["series_id"]: row.to_dict() for _, row in weights.iterrows()}


def validate_main_dataset(df_main):
    required_columns = {"region", "commodity_name", "month", "price"}
    missing = sorted(required_columns.difference(df_main.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def prepare_base_panel(main_df=None):
    if main_df is None:
        df_main = pd.read_csv(ROOT / "data" / "main" / "Combined Main Dataset.csv", low_memory=False)
    else:
        df_main = main_df.copy()

    validate_main_dataset(df_main)
    df_main["month"] = pd.to_datetime(df_main["month"], errors="coerce")
    df_main["price"] = pd.to_numeric(df_main["price"], errors="coerce")

    ex_diesel_raw = pd.read_csv(ROOT / "data" / "exogenous" / "Diesel Price.csv")
    ex_food_index_raw = pd.read_csv(
        ROOT / "data" / "exogenous" / "Monthly food price estimates by product and market (2007-2025).csv"
    )
    ex_weather_1 = pd.read_csv(
        ROOT / "data" / "exogenous" / "philippines_weather_cdd_r50mm_hd35_monthly_2000-2023.csv"
    )
    ex_weather_2 = pd.read_csv(
        ROOT / "data" / "exogenous" / "philippines_weather_era5_monthly_2000-2023.csv"
    )

    ex_weather_raw = ex_weather_1.drop(columns=["DaysRainAbove50mm"]).merge(
        ex_weather_2,
        on="Date",
        how="inner",
    )

    ex_diesel_monthly = ex_diesel_raw.copy()
    ex_diesel_monthly["month"] = pd.to_datetime(ex_diesel_monthly["Month"], format="%y-%b", errors="coerce")
    ex_diesel_monthly["month"] = ex_diesel_monthly["month"].dt.to_period("M").dt.to_timestamp()
    ex_diesel_monthly["diesel_price"] = pd.to_numeric(ex_diesel_monthly["Price"], errors="coerce")
    ex_diesel_monthly = ex_diesel_monthly[["month", "diesel_price"]].dropna().sort_values("month").reset_index(drop=True)

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

    if not valid_series:
        raise ValueError("No valid series remained after interpolation and minimum-length filtering.")

    df_main_filtered = pd.concat(valid_series, ignore_index=True)
    df_main_filtered = df_main_filtered.sort_values(["region", "commodity_name", "month"]).reset_index(drop=True)

    regional_panel = df_main_filtered.merge(macro_external_monthly, on="month", how="left")
    regional_preprocessed_full = hf.add_basic_price_features(regional_panel)
    regional_preprocessed_full, regional_exogenous_feature_columns = hf.add_exogenous_lags(
        regional_preprocessed_full,
        drivers=EXOG_DRIVER_COLUMNS,
    )

    regional_series_manifest, regional_preprocessed_panel = hf.build_series_manifest(regional_preprocessed_full)
    all_eligible_series_manifest = regional_series_manifest.copy()

    if len(all_eligible_series_manifest) >= MODEL_SERIES_CAP:
        modeling_series_cap = MODEL_SERIES_CAP
    elif len(all_eligible_series_manifest) >= MODEL_SERIES_CAP_FALLBACK:
        modeling_series_cap = MODEL_SERIES_CAP_FALLBACK
    else:
        modeling_series_cap = len(all_eligible_series_manifest)

    eligible_series_manifest = (
        hf.select_series_manifest_balanced(
            all_eligible_series_manifest,
            modeling_series_cap,
            group_col="region",
        )
    )

    eligible_series_ids = set(eligible_series_manifest["series_id"])
    eligible_panel = regional_preprocessed_panel.loc[
        regional_preprocessed_panel["series_id"].isin(eligible_series_ids)
    ].copy()

    return eligible_panel, eligible_series_manifest, regional_exogenous_feature_columns


def reconstruct_price_from_yoy(price_frame, forecast_month, yoy_value):
    forecast_month = pd.Timestamp(forecast_month).to_period("M").to_timestamp()
    reference_month = forecast_month - pd.DateOffset(years=1)
    reference_price = price_frame.loc[price_frame["month"] == reference_month, "price"].dropna()
    if reference_price.empty or pd.isna(yoy_value):
        return np.nan
    return float(reference_price.iloc[-1] * (1.0 + (float(yoy_value) / 100.0)))


def extend_series_for_future(series_df, driver_cols=None):
    driver_cols = driver_cols or []
    working = series_df.copy().sort_values("month").reset_index(drop=True)
    last_row = working.iloc[-1]
    next_month = pd.Timestamp(last_row["month"]) + pd.offsets.MonthBegin(1)
    new_row = {column: np.nan for column in working.columns}

    for column in ["series_id", "region", "commodity_name"]:
        if column in working.columns:
            new_row[column] = last_row.get(column)

    new_row["month"] = next_month
    for driver in driver_cols:
        if driver in working.columns:
            last_known = working[driver].dropna()
            if not last_known.empty:
                new_row[driver] = float(last_known.iloc[-1])

    return pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)


def forecast_sarima_future(series_df, settings_row, horizon):
    order = parse_literal(settings_row.get("sarima_order"), fallback=None)
    if order is None:
        order = settings_row.get("order")
    seasonal_order = parse_literal(settings_row.get("sarima_seasonal_order"), fallback=None)
    if seasonal_order is None:
        seasonal_order = settings_row.get("seasonal_order")
    trend = settings_row.get("sarima_trend", settings_row.get("trend", "n"))
    if order is None or seasonal_order is None:
        return pd.DataFrame()

    history = series_df.sort_values("month").reset_index(drop=True).copy()
    if TARGET_COL not in history.columns:
        history = hf.add_basic_price_features(history)
    target_history = history.loc[history[TARGET_COL].notna(), TARGET_COL]
    if len(target_history) < MIN_TRAIN_OBS_SARIMA:
        return pd.DataFrame()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = SARIMAX(
                target_history,
                order=tuple(order),
                seasonal_order=tuple(seasonal_order),
                trend=trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(disp=False, maxiter=SARIMA_MAXITER)
        yoy_forecasts = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    except Exception:
        return pd.DataFrame()

    price_history = history[["month", "price"]].copy()
    last_month = pd.Timestamp(history["month"].max()).to_period("M").to_timestamp()
    rows = []

    for step, yoy_value in enumerate(yoy_forecasts, start=1):
        forecast_month = last_month + pd.offsets.MonthBegin(step)
        forecast_price = reconstruct_price_from_yoy(price_history, forecast_month, yoy_value)
        rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": forecast_month,
            "predicted_yoy": float(yoy_value),
            "predicted_price": forecast_price,
            "model": "SARIMA",
        })
        price_history = pd.concat(
            [price_history, pd.DataFrame([{"month": forecast_month, "price": forecast_price}])],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def forecast_svr_future(series_df, settings_row, feature_cols, horizon):
    best_params = parse_literal(settings_row.get("svr_best_params"), fallback={}) or {}
    if not best_params:
        best_params = settings_row.get("best_params", {}) or {}
    featured_history = hf.add_basic_price_features(series_df)
    model_train = featured_history.loc[
        featured_history[feature_cols + [TARGET_COL]].notna().all(axis=1),
        feature_cols + [TARGET_COL],
    ].copy()
    if len(model_train) < MIN_TRAIN_OBS_ML:
        return pd.DataFrame()

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
    except Exception:
        return pd.DataFrame()

    working = series_df.copy().sort_values("month").reset_index(drop=True)
    rows = []
    for _ in range(horizon):
        working = extend_series_for_future(working)
        featured = hf.add_basic_price_features(working)
        predict_row = featured.iloc[-1]
        history_target = featured.loc[featured[TARGET_COL].notna(), ["month", TARGET_COL]].copy()
        fallback = hf.seasonal_naive_forecast(history_target, predict_row["month"], TARGET_COL) if not history_target.empty else np.nan
        if pd.isna(fallback) and not history_target.empty:
            fallback = float(history_target[TARGET_COL].iloc[-1])

        if predict_row[feature_cols].notna().all():
            try:
                predicted_yoy = float(model.predict(pd.DataFrame([predict_row[feature_cols]], columns=feature_cols))[0])
            except Exception:
                predicted_yoy = fallback
        else:
            predicted_yoy = fallback

        predicted_price = reconstruct_price_from_yoy(working[["month", "price"]], predict_row["month"], predicted_yoy)
        working.loc[working.index[-1], "price"] = predicted_price
        rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": predict_row["month"],
            "predicted_yoy": float(predicted_yoy) if pd.notna(predicted_yoy) else np.nan,
            "predicted_price": predicted_price,
            "model": "SVR",
        })
    return pd.DataFrame(rows)


def forecast_lightgbm_future(series_df, settings_row, feature_cols, exog_cols, horizon):
    best_params = parse_literal(settings_row.get("lightgbm_params"), fallback={}) or {}
    if not best_params:
        best_params = settings_row.get("best_params", {}) or {}
    featured_history = hf.add_basic_price_features(series_df)
    featured_history, _ = hf.add_exogenous_lags(featured_history, drivers=EXOG_DRIVER_COLUMNS)

    for driver in EXOG_DRIVER_COLUMNS:
        if driver in featured_history.columns:
            featured_history[driver] = featured_history[driver].ffill().bfill()

    model_train = featured_history.loc[
        featured_history[feature_cols + [TARGET_COL]].notna().all(axis=1),
        feature_cols + [TARGET_COL],
    ].copy()
    if len(model_train) < MIN_TRAIN_OBS_ML:
        return pd.DataFrame()

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
    except Exception:
        return pd.DataFrame()

    working = series_df.copy().sort_values("month").reset_index(drop=True)
    rows = []
    for _ in range(horizon):
        working = extend_series_for_future(working, driver_cols=EXOG_DRIVER_COLUMNS)
        featured = hf.add_basic_price_features(working)
        featured, _ = hf.add_exogenous_lags(featured, drivers=EXOG_DRIVER_COLUMNS)
        predict_row = featured.iloc[-1].copy()
        predict_row = hf.fill_lightgbm_exog_from_history(featured.iloc[:-1], predict_row, exog_cols)

        history_target = featured.loc[featured[TARGET_COL].notna(), ["month", TARGET_COL]].copy()
        fallback = hf.seasonal_naive_forecast(history_target, predict_row["month"], TARGET_COL) if not history_target.empty else np.nan
        if pd.isna(fallback) and not history_target.empty:
            fallback = float(history_target[TARGET_COL].iloc[-1])

        if predict_row[feature_cols].notna().all():
            try:
                predicted_yoy = float(model.predict(pd.DataFrame([predict_row[feature_cols]], columns=feature_cols))[0])
            except Exception:
                predicted_yoy = fallback
        else:
            predicted_yoy = fallback

        predicted_price = reconstruct_price_from_yoy(working[["month", "price"]], predict_row["month"], predicted_yoy)
        working.loc[working.index[-1], "price"] = predicted_price
        rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": predict_row["month"],
            "predicted_yoy": float(predicted_yoy) if pd.notna(predicted_yoy) else np.nan,
            "predicted_price": predicted_price,
            "model": "LightGBM",
        })
    return pd.DataFrame(rows)


def get_holdout_positions(df, min_train_obs):
    valid_positions = df.index[df[TARGET_COL].notna()].tolist()
    if len(valid_positions) < min_train_obs + 1:
        return []

    holdout_count = max(1, int(np.ceil(len(valid_positions) * HOLDOUT_RATIO)))
    holdout_count = min(holdout_count, len(valid_positions) - min_train_obs)
    if holdout_count <= 0:
        return []

    return valid_positions[-holdout_count:]


def generate_sarima_holdout(series_df, settings_row):
    order = parse_literal(settings_row.get("sarima_order"), fallback=None)
    if order is None:
        order = settings_row.get("order")
    seasonal_order = parse_literal(settings_row.get("sarima_seasonal_order"), fallback=None)
    if seasonal_order is None:
        seasonal_order = settings_row.get("seasonal_order")
    trend = settings_row.get("sarima_trend", settings_row.get("trend", "n"))
    if order is None or seasonal_order is None:
        return pd.DataFrame()

    df = series_df.sort_values("month").reset_index(drop=True).copy()
    test_positions = get_holdout_positions(df, MIN_TRAIN_OBS_SARIMA)
    if not test_positions:
        return pd.DataFrame()

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
        seasonal_pred = hf.seasonal_naive_forecast(tuning_history, row["month"], TARGET_COL)
        if pd.isna(seasonal_pred):
            seasonal_pred = naive_pred

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sarima_fit = SARIMAX(
                    tuning_history[TARGET_COL],
                    order=tuple(order),
                    seasonal_order=tuple(seasonal_order),
                    trend=trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=SARIMA_MAXITER)

            sarima_pred = float(np.asarray(sarima_fit.forecast(steps=1))[0])
            if not np.isfinite(sarima_pred) or abs(sarima_pred) > 1e6:
                sarima_pred = naive_pred
        except Exception:
            sarima_pred = naive_pred

        prediction_rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": row["month"],
            "actual": float(actual),
            "naive_pred": float(naive_pred),
            "seasonal_naive_pred": float(seasonal_pred),
            "sarima_pred": float(sarima_pred),
        })

    return pd.DataFrame(prediction_rows)


def generate_svr_holdout(series_df, settings_row):
    feature_cols = settings_row.get("feature_columns", []) or []
    best_params = settings_row.get("best_params", {}) or {}
    if not feature_cols:
        return pd.DataFrame()

    df = series_df.sort_values("month").reset_index(drop=True).copy()
    test_positions = get_holdout_positions(df, MIN_TRAIN_OBS_ML)
    if not test_positions:
        return pd.DataFrame()

    prediction_rows = []
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

        model_train = train_block.loc[
            train_block[feature_cols + [TARGET_COL]].notna().all(axis=1),
            feature_cols + [TARGET_COL],
        ].copy()
        feature_ready = bool(feature_cols) and row[feature_cols].notna().all()

        if len(model_train) >= MIN_TRAIN_OBS_ML and feature_ready:
            try:
                final_model = Pipeline([
                    ("scaler", RobustScaler()),
                    ("svr", SVR(
                        kernel="rbf",
                        C=best_params.get("svr__C", 1.0),
                        epsilon=best_params.get("svr__epsilon", 0.1),
                        gamma=best_params.get("svr__gamma", "scale"),
                    )),
                ])
                final_model.fit(model_train[feature_cols], model_train[TARGET_COL])
                svr_pred = float(final_model.predict(pd.DataFrame([row[feature_cols].values], columns=feature_cols))[0])
            except Exception:
                svr_pred = naive_pred
        else:
            svr_pred = naive_pred

        prediction_rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": row["month"],
            "actual": float(actual),
            "naive_pred": float(naive_pred),
            "seasonal_naive_pred": float(seasonal_pred),
            "svr_pred": float(svr_pred),
        })

    return pd.DataFrame(prediction_rows)


def generate_lightgbm_holdout(series_df, settings_row):
    feature_cols = settings_row.get("feature_columns", []) or []
    exog_cols = settings_row.get("exogenous_feature_columns", []) or []
    best_params = settings_row.get("best_params", {}) or {}
    if not feature_cols:
        return pd.DataFrame()

    df = series_df.sort_values("month").reset_index(drop=True).copy()
    test_positions = get_holdout_positions(df, MIN_TRAIN_OBS_ML)
    if not test_positions:
        return pd.DataFrame()

    prediction_rows = []
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

        model_train = train_block.copy()
        for col in exog_cols:
            if col in model_train.columns:
                model_train[col] = model_train[col].ffill().bfill()

        model_train = model_train.loc[
            model_train[feature_cols + [TARGET_COL]].notna().all(axis=1),
            feature_cols + [TARGET_COL],
        ].copy()
        feature_ready = bool(feature_cols) and row[feature_cols].notna().all()

        if len(model_train) >= MIN_TRAIN_OBS_ML and feature_ready:
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
                lightgbm_pred = float(model.predict(pd.DataFrame([row[feature_cols].values], columns=feature_cols))[0])
            except Exception:
                lightgbm_pred = naive_pred
        else:
            lightgbm_pred = naive_pred

        prediction_rows.append({
            "series_id": settings_row["series_id"],
            "region": settings_row["region"],
            "commodity_name": settings_row["commodity_name"],
            "month": row["month"],
            "actual": float(actual),
            "naive_pred": float(naive_pred),
            "seasonal_naive_pred": float(seasonal_pred),
            "lightgbm_pred": float(lightgbm_pred),
        })

    return pd.DataFrame(prediction_rows)


def inverse_rmse_weight(value):
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        return 0.0
    return 1.0 / float(value)


def normalize_ensemble_components(weight_row):
    raw_weights = {
        "SARIMA": inverse_rmse_weight(weight_row.get("sarima_inner_rmse")),
        "SVR": inverse_rmse_weight(weight_row.get("svr_inner_rmse")),
        "LightGBM": inverse_rmse_weight(weight_row.get("lightgbm_inner_rmse")),
    }
    positive_total = sum(
        weight for weight in raw_weights.values()
        if np.isfinite(weight) and weight > 0
    )
    normalized_weights = {model_name: np.nan for model_name in raw_weights}

    if positive_total > 0:
        for model_name, raw_weight in raw_weights.items():
            if np.isfinite(raw_weight) and raw_weight > 0:
                normalized_weights[model_name] = raw_weight / positive_total

    ranked_models = [
        (model_name, normalized_weights[model_name])
        for model_name in ["SARIMA", "SVR", "LightGBM"]
        if pd.notna(normalized_weights[model_name])
    ]
    ranked_models.sort(key=lambda item: (-item[1], item[0]))
    dominant_model = ranked_models[0][0] if ranked_models else None
    dominant_weight = ranked_models[0][1] if ranked_models else np.nan

    return {
        "sarima_weight": normalized_weights["SARIMA"],
        "svr_weight": normalized_weights["SVR"],
        "lightgbm_weight": normalized_weights["LightGBM"],
        "dominant_ensemble_model": dominant_model,
        "dominant_ensemble_weight": dominant_weight,
    }


def build_series_profiles(series_options, series_metrics_export, ensemble_weight_lookup):
    profiles = (
        series_options[["series_id", "region", "commodity_name"]]
        .drop_duplicates()
        .copy()
    )

    if not series_metrics_export.empty:
        best_holdout = (
            series_metrics_export.sort_values(["series_id", "rmse", "model"])
            .groupby("series_id", as_index=False)
            .first()[["series_id", "model", "rmse"]]
            .rename(columns={
                "model": "best_holdout_model",
                "rmse": "best_holdout_rmse",
            })
        )
        profiles = profiles.merge(best_holdout, on="series_id", how="left")
    else:
        profiles["best_holdout_model"] = None
        profiles["best_holdout_rmse"] = np.nan

    weight_rows = []
    for series_id in profiles["series_id"]:
        weight_row = ensemble_weight_lookup.get(series_id, {})
        normalized = normalize_ensemble_components(weight_row)
        weight_rows.append({
            "series_id": series_id,
            "sarima_inner_rmse": weight_row.get("sarima_inner_rmse"),
            "svr_inner_rmse": weight_row.get("svr_inner_rmse"),
            "lightgbm_inner_rmse": weight_row.get("lightgbm_inner_rmse"),
            "sarima_weight": normalized["sarima_weight"],
            "svr_weight": normalized["svr_weight"],
            "lightgbm_weight": normalized["lightgbm_weight"],
            "dominant_ensemble_model": normalized["dominant_ensemble_model"],
            "dominant_ensemble_weight": normalized["dominant_ensemble_weight"],
        })

    if weight_rows:
        profiles = profiles.merge(pd.DataFrame(weight_rows), on="series_id", how="left")

    return profiles.sort_values(["commodity_name", "region"]).reset_index(drop=True)


def compute_series_normalized_metrics(holdout_frame):
    rows = []

    for (series_id, region, commodity_name, model_name), part in holdout_frame.groupby(
        ["series_id", "region", "commodity_name", "model"]
    ):
        valid = part.loc[
            part["actual_yoy"].notna() & part["predicted_yoy"].notna(),
            ["actual_yoy", "predicted_yoy"],
        ].copy()
        if valid.empty:
            continue

        actual_range = float(valid["actual_yoy"].max() - valid["actual_yoy"].min())
        current_rmse = hf.rmse(valid["actual_yoy"], valid["predicted_yoy"])
        current_mae = hf.mae(valid["actual_yoy"], valid["predicted_yoy"])

        rows.append({
            "series_id": series_id,
            "region": region,
            "commodity_name": commodity_name,
            "model": model_name,
            "holdout_actual_range_yoy": actual_range,
            "nrmse_range": current_rmse / actual_range if actual_range > 0 else np.nan,
            "nmae_range": current_mae / actual_range if actual_range > 0 else np.nan,
        })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics

    metrics["best_series_win"] = False
    for series_id, part in metrics.groupby("series_id"):
        valid_part = part.loc[part["nrmse_range"].notna()].copy()
        if valid_part.empty:
            continue
        winner_idx = valid_part.sort_values(["nrmse_range", "model"], ascending=[True, True]).index[0]
        metrics.loc[winner_idx, "best_series_win"] = True

    return metrics


def compute_model_consistency_summary(series_metrics_frame):
    if series_metrics_frame.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "series_evaluated",
                "mean_series_nrmse",
                "median_series_nrmse",
                "best_series_wins",
            ]
        )

    summary = (
        series_metrics_frame.groupby("model", as_index=False)
        .agg(
            series_evaluated=("series_id", "nunique"),
            mean_series_nrmse=("nrmse_range", "mean"),
            median_series_nrmse=("nrmse_range", "median"),
            best_series_wins=("best_series_win", lambda values: int(pd.Series(values).fillna(False).sum())),
        )
        .sort_values(["mean_series_nrmse", "median_series_nrmse", "model"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return summary


def build_dashboard_payload(main_df=None):
    eligible_panel, eligible_series_manifest, regional_exogenous_feature_columns = prepare_base_panel(main_df=main_df)

    sarima_settings_lookup = load_artifact_payloads("SARIMA", "sarima")
    svr_settings_lookup = load_artifact_payloads("SVR", "svr")
    lightgbm_settings_lookup = load_artifact_payloads("LightGBM", "lightgbm")
    ensemble_weight_lookup = load_ensemble_weight_lookup()

    sarima_parts = []
    svr_parts = []
    lightgbm_parts = []

    for series_id, part in eligible_panel.groupby("series_id"):
        part = part.sort_values("month").reset_index(drop=True).copy()

        if series_id in sarima_settings_lookup:
            sarima_part = generate_sarima_holdout(part, sarima_settings_lookup[series_id])
            if not sarima_part.empty:
                sarima_parts.append(sarima_part)

        if series_id in svr_settings_lookup:
            svr_part = generate_svr_holdout(part, svr_settings_lookup[series_id])
            if not svr_part.empty:
                svr_parts.append(svr_part)

        if series_id in lightgbm_settings_lookup:
            lightgbm_part = generate_lightgbm_holdout(part, lightgbm_settings_lookup[series_id])
            if not lightgbm_part.empty:
                lightgbm_parts.append(lightgbm_part)

    sarima_predictions = pd.concat(sarima_parts, ignore_index=True) if sarima_parts else pd.DataFrame(
        columns=["series_id", "region", "commodity_name", "month", "actual", "naive_pred", "seasonal_naive_pred", "sarima_pred"]
    )
    svr_predictions = pd.concat(svr_parts, ignore_index=True) if svr_parts else pd.DataFrame(
        columns=["series_id", "region", "commodity_name", "month", "actual", "naive_pred", "seasonal_naive_pred", "svr_pred"]
    )
    lightgbm_predictions = pd.concat(lightgbm_parts, ignore_index=True) if lightgbm_parts else pd.DataFrame(
        columns=["series_id", "region", "commodity_name", "month", "actual", "naive_pred", "seasonal_naive_pred", "lightgbm_pred"]
    )

    ensemble_predictions = sarima_predictions.merge(
        svr_predictions[["series_id", "region", "commodity_name", "month", "svr_pred"]],
        on=["series_id", "region", "commodity_name", "month"],
        how="inner",
    ).merge(
        lightgbm_predictions[["series_id", "region", "commodity_name", "month", "lightgbm_pred"]],
        on=["series_id", "region", "commodity_name", "month"],
        how="inner",
    )

    if not ensemble_predictions.empty:
        weights_frame = pd.DataFrame(list(ensemble_weight_lookup.values()))
        ensemble_predictions = ensemble_predictions.merge(weights_frame, on="series_id", how="left")

        weighted_preds = []
        for _, row in ensemble_predictions.iterrows():
            preds = np.asarray([row["sarima_pred"], row["svr_pred"], row["lightgbm_pred"]], dtype=float)
            weights = np.asarray(
                [
                    inverse_rmse_weight(row.get("sarima_inner_rmse")),
                    inverse_rmse_weight(row.get("svr_inner_rmse")),
                    inverse_rmse_weight(row.get("lightgbm_inner_rmse")),
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

    history_export = (
        eligible_panel[["series_id", "region", "commodity_name", "month", "price", TARGET_COL]]
        .rename(columns={"price": "actual_price", TARGET_COL: "actual_yoy"})
        .sort_values(["region", "commodity_name", "month"])
        .reset_index(drop=True)
    )
    history_export["month"] = pd.to_datetime(history_export["month"]).dt.strftime("%Y-%m-%d")

    holdout_base = ensemble_predictions.merge(
        eligible_panel[["series_id", "month", "price", "price_lag_12"]],
        on=["series_id", "month"],
        how="left",
    )
    holdout_base["month"] = pd.to_datetime(holdout_base["month"], errors="coerce")

    holdout_model_columns = {
        "SARIMA": "sarima_pred",
        "SVR": "svr_pred",
        "LightGBM": "lightgbm_pred",
        "Weighted Ensemble": "weighted_ensemble_pred",
    }

    holdout_parts = []
    for model_name, prediction_column in holdout_model_columns.items():
        part = holdout_base[
            ["series_id", "region", "commodity_name", "month", "actual", prediction_column, "price", "price_lag_12"]
        ].copy()
        part = part.rename(columns={
            "actual": "actual_yoy",
            prediction_column: "predicted_yoy",
            "price": "actual_price",
        })
        part["model"] = model_name
        part["predicted_price"] = part["price_lag_12"] * (1.0 + (part["predicted_yoy"] / 100.0))
        part["abs_error_yoy"] = (part["actual_yoy"] - part["predicted_yoy"]).abs()
        part["abs_error_price"] = (part["actual_price"] - part["predicted_price"]).abs()
        part["pct_error_price"] = np.where(
            part["actual_price"].abs() > 0,
            (part["abs_error_price"] / part["actual_price"].abs()) * 100.0,
            np.nan,
        )
        part["phase"] = "holdout"
        holdout_parts.append(part)

    holdout_export = pd.concat(holdout_parts, ignore_index=True) if holdout_parts else pd.DataFrame(
        columns=[
            "series_id", "region", "commodity_name", "month", "actual_yoy", "predicted_yoy",
            "actual_price", "predicted_price", "abs_error_yoy", "abs_error_price", "pct_error_price", "phase", "model"
        ]
    )
    if not holdout_export.empty:
        holdout_export["month"] = pd.to_datetime(holdout_export["month"]).dt.strftime("%Y-%m-%d")
        holdout_export = holdout_export.sort_values(["region", "commodity_name", "model", "month"]).reset_index(drop=True)

    prediction_pairs = [
        ("Naive", "naive_pred"),
        ("Seasonal Naive", "seasonal_naive_pred"),
        ("SARIMA", "sarima_pred"),
        ("SVR", "svr_pred"),
        ("LightGBM", "lightgbm_pred"),
        ("Weighted Ensemble", "weighted_ensemble_pred"),
    ]
    series_metrics_export = hf.compute_series_metrics(ensemble_predictions, prediction_pairs)
    global_metrics_export = hf.compute_metrics_table(ensemble_predictions, prediction_pairs)
    if series_metrics_export.empty:
        series_metrics_export = pd.DataFrame(
            columns=["series_id", "region", "commodity_name", "model", "rmse", "mae", "r2", "rows_evaluated"]
        )
    else:
        series_metrics_export = series_metrics_export.loc[series_metrics_export["model"].isin(APP_MODELS)].copy()
    if global_metrics_export.empty:
        global_metrics_export = pd.DataFrame(columns=["model", "rmse", "mae", "r2", "rows_evaluated"])
    else:
        global_metrics_export = global_metrics_export.loc[global_metrics_export["model"].isin(APP_MODELS)].copy()

    lightgbm_exog_candidates = [feature for feature in regional_exogenous_feature_columns if feature in eligible_panel.columns]
    if lightgbm_exog_candidates:
        lightgbm_exog_coverage = eligible_panel[lightgbm_exog_candidates].notna().mean().sort_values(ascending=False)
        lightgbm_exog_features = lightgbm_exog_coverage.loc[lightgbm_exog_coverage >= 0.60].index.tolist()[:8]
    else:
        lightgbm_exog_features = []
    lightgbm_features = [feature for feature in LIGHTGBM_BASE_FEATURES if feature in eligible_panel.columns] + lightgbm_exog_features
    svr_features = [feature for feature in SVR_BASE_FEATURES if feature in eligible_panel.columns]

    future_parts = []
    for series_id in sorted(eligible_panel["series_id"].unique()):
        if series_id not in sarima_settings_lookup and series_id not in svr_settings_lookup and series_id not in lightgbm_settings_lookup:
            continue

        series_frame = eligible_panel.loc[
            eligible_panel["series_id"] == series_id,
            ["series_id", "region", "commodity_name", "month", "price"] + EXOG_DRIVER_COLUMNS
        ].copy()
        model_future_frames = []

        if series_id in sarima_settings_lookup:
            sarima_future = forecast_sarima_future(
                series_frame[["series_id", "region", "commodity_name", "month", "price"]],
                sarima_settings_lookup[series_id],
                FORECAST_MONTHS,
            )
            if not sarima_future.empty:
                model_future_frames.append(sarima_future)

        if series_id in svr_settings_lookup:
            svr_feature_columns = svr_settings_lookup[series_id].get("feature_columns", svr_features)
            svr_future = forecast_svr_future(
                series_frame[["series_id", "region", "commodity_name", "month", "price"]],
                svr_settings_lookup[series_id],
                svr_feature_columns,
                FORECAST_MONTHS,
            )
            if not svr_future.empty:
                model_future_frames.append(svr_future)

        if series_id in lightgbm_settings_lookup:
            lightgbm_feature_columns = lightgbm_settings_lookup[series_id].get("feature_columns", lightgbm_features)
            lightgbm_exog_columns = lightgbm_settings_lookup[series_id].get(
                "exogenous_feature_columns",
                lightgbm_exog_features,
            )
            lightgbm_future = forecast_lightgbm_future(
                series_frame,
                lightgbm_settings_lookup[series_id],
                lightgbm_feature_columns,
                lightgbm_exog_columns,
                FORECAST_MONTHS,
            )
            if not lightgbm_future.empty:
                model_future_frames.append(lightgbm_future)

        if not model_future_frames:
            continue

        series_future = pd.concat(model_future_frames, ignore_index=True)
        future_parts.append(series_future)

        weight_row = ensemble_weight_lookup.get(series_id, {})
        weight_map = {
            "SARIMA": inverse_rmse_weight(weight_row.get("sarima_inner_rmse")),
            "SVR": inverse_rmse_weight(weight_row.get("svr_inner_rmse")),
            "LightGBM": inverse_rmse_weight(weight_row.get("lightgbm_inner_rmse")),
        }
        ensemble_rows = []
        for forecast_month, month_part in series_future.groupby("month"):
            yoy_values = []
            price_values = []
            weights = []
            for model_name in ["SARIMA", "SVR", "LightGBM"]:
                model_row = month_part.loc[month_part["model"] == model_name]
                if model_row.empty:
                    continue
                yoy_values.append(float(model_row["predicted_yoy"].iloc[0]))
                price_values.append(float(model_row["predicted_price"].iloc[0]) if pd.notna(model_row["predicted_price"].iloc[0]) else np.nan)
                weights.append(weight_map.get(model_name, 0.0))

            if not yoy_values:
                continue

            yoy_values = np.asarray(yoy_values, dtype=float)
            price_values = np.asarray(price_values, dtype=float)
            weights = np.asarray(weights, dtype=float)

            if np.isfinite(weights).any() and weights.sum() > 0:
                ensemble_yoy = float(np.average(yoy_values, weights=weights))
                finite_prices = np.isfinite(price_values)
                ensemble_price = float(np.average(price_values[finite_prices], weights=weights[finite_prices])) if finite_prices.any() and weights[finite_prices].sum() > 0 else np.nan
            else:
                ensemble_yoy = float(np.nanmean(yoy_values))
                ensemble_price = float(np.nanmean(price_values)) if np.isfinite(price_values).any() else np.nan

            ensemble_rows.append({
                "series_id": series_id,
                "region": series_frame["region"].iloc[0],
                "commodity_name": series_frame["commodity_name"].iloc[0],
                "month": forecast_month,
                "predicted_yoy": ensemble_yoy,
                "predicted_price": ensemble_price,
                "model": "Weighted Ensemble",
            })

        if ensemble_rows:
            future_parts.append(pd.DataFrame(ensemble_rows))

    future_export = pd.concat(future_parts, ignore_index=True) if future_parts else pd.DataFrame(
        columns=["series_id", "region", "commodity_name", "month", "predicted_yoy", "predicted_price", "model"]
    )
    future_export["phase"] = "future"
    if not future_export.empty:
        future_export["month"] = pd.to_datetime(future_export["month"]).dt.strftime("%Y-%m-%d")
        future_export = future_export.sort_values(["region", "commodity_name", "model", "month"]).reset_index(drop=True)

    normalized_series_metrics = compute_series_normalized_metrics(holdout_export)
    model_consistency_summary = compute_model_consistency_summary(normalized_series_metrics)
    if not normalized_series_metrics.empty:
        series_metrics_export = series_metrics_export.merge(
            normalized_series_metrics[
                [
                    "series_id",
                    "model",
                    "holdout_actual_range_yoy",
                    "nrmse_range",
                    "nmae_range",
                    "best_series_win",
                ]
            ],
            on=["series_id", "model"],
            how="left",
        )

    series_options = (
        eligible_series_manifest[["series_id", "region", "commodity_name"]]
        .drop_duplicates()
        .sort_values(["commodity_name", "region"])
        .reset_index(drop=True)
    )
    series_profiles = build_series_profiles(series_options, series_metrics_export, ensemble_weight_lookup)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "forecast_horizon_months": FORECAST_MONTHS,
        "target_column": TARGET_COL,
        "target_label": TARGET_LABEL,
        "models": APP_MODELS,
        "series_options": series_options.where(pd.notna(series_options), None).to_dict(orient="records"),
        "series_profiles": series_profiles.where(pd.notna(series_profiles), None).to_dict(orient="records"),
        "history": history_export.where(pd.notna(history_export), None).to_dict(orient="records"),
        "holdout": holdout_export.where(pd.notna(holdout_export), None).to_dict(orient="records"),
        "future": future_export.where(pd.notna(future_export), None).to_dict(orient="records"),
        "series_metrics": series_metrics_export.where(pd.notna(series_metrics_export), None).to_dict(orient="records"),
    }

    export_frame = pd.concat(
        [
            holdout_export.copy(),
            future_export.copy(),
        ],
        ignore_index=True,
    )

    return payload, export_frame


def write_dashboard_payload(payload, output_path):
    with Path(output_path).open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def export_results_csv(frame, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main():
    payload, export_frame = build_dashboard_payload()
    write_dashboard_payload(payload, WEBAPP_DATA_DIR / "dashboard.json")
    export_results_csv(export_frame, WEBAPP_DATA_DIR / "dashboard_results.csv")
    print(f"Exported webapp payload to {WEBAPP_DATA_DIR / 'dashboard.json'}")


if __name__ == "__main__":
    main()
