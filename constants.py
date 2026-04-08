# Granger Causality Test
MAX_LAG = 6
MIN_OBS = 36
ALPHA = 0.05
SHORTLIST_MIN_OVERLAP = 60
TRANSFORM_MODES = ["diff", "mom_pct"]
GLOBAL_DRIVER_MIN_SHORTLISTED_SERIES = 8

DRIVER_COLUMNS = [
    "diesel_price",
    "food_price_index",
    "max_consecutive_dry_days",
    "days_above_35c",
    "precipitation_mm",
    "days_rain_above_50mm_extreme",
    "avg_mean_temp_c",
]

# Panel filtering
WINDOW_MIN_MONTHS = 36
WINDOW_MAX_MONTHS = 60
# Set to -1 to use all eligible series.
MODEL_SERIES_CAP = 100
MODEL_SERIES_CAP_FALLBACK = 50

# Target setup
TARGET_COL = "yoy_inflation"
TARGET_LABEL = "YoY Inflation (%)"
MIN_TRAIN_TARGET_ROWS = 24
FORECAST_HORIZON = 1
ROLLING_STEP = 1
SEASONAL_PERIOD = 12

# SARIMA
HOLDOUT_RATIO = 0.20
MIN_TRAIN_OBS_SARIMA = 24
INNER_FOLDS_SARIMA = 5
SARIMA_P_VALUES = [0, 1, 2]
SARIMA_Q_VALUES = [0, 1, 2]
SARIMA_P_SEASONAL_VALUES = [0, 1]
SARIMA_Q_SEASONAL_VALUES = [0, 1]
SARIMA_TREND_VALUES = ["n", "c"]
SARIMA_MAXITER = 300

# SVR
SVR_BASE_FEATURES = [
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
    "yoy_lag1_x_roll3",
    "yoy_lag1_x_roll6",
    "log_price_lag_1",
    "log_price_lag_3",
    "price_index_lag_1",
    "price_index_roll_mean_3",
    "price_index_roll_std_3",
    "month_sin",
    "month_cos",
]
SVR_PARAM_GRID = {
    "svr__C": [0.3, 1.0, 3.0, 10.0],
    "svr__epsilon": [0.05, 0.1, 0.2, 0.3],
    "svr__gamma": ["scale", 0.01],
}
MIN_TRAIN_OBS_ML = MIN_TRAIN_OBS_SARIMA
INNER_FOLDS_ML = INNER_FOLDS_SARIMA

# LightGBM
LIGHTGBM_BASE_FEATURES = [
    "yoy_lag_1",
    "yoy_lag_2",
    "yoy_lag_3",
    "yoy_lag_6",
    "yoy_roll_mean_3",
    "yoy_roll_mean_6",
    "yoy_roll_std_3",
    "yoy_roll_std_6",
    "yoy_acceleration",
    "log_price_lag_1",
    "log_price_lag_3",
    "log_price_lag_12",
    "log_price_yoy_lag_1",
    "log_price_yoy_lag_3",
    "yoy_lag1_x_roll3",
    "yoy_lag1_x_roll6",
    "price_index_lag_1",
    "price_index_lag_3",
    "price_index_roll_mean_3",
    "price_index_roll_std_3",
    "price_index_gap_from_roll3",
    "price_index_momentum_3",
    "month_sin",
    "month_cos",
    "post_2020_break",
    "covid_shock_window",
]
LIGHTGBM_TRIALS = 12
LIGHTGBM_RANDOM_STATE = 42
