# Webapp

This directory contains a static dashboard for the thesis forecasts.

## Expected data

Run these in order:

```powershell
python main.py
python webapp_export.py
```

`main.py` produces the model tables, and `webapp_export.py` converts them into:

- `webapp/data/dashboard.json`

The dashboard reads that file directly.

## Run locally

From the project root:

```powershell
python webapp/serve.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Features

- Filter by commodity and region
- Switch between SARIMA, SVR, LightGBM, and Weighted Ensemble
- Toggle actual-only, forecast-only, or actual-vs-forecast
- Hover to inspect price and YoY values
- Click a point to inspect actual, forecast, and error details
- View per-series and global holdout metrics beside the chart
