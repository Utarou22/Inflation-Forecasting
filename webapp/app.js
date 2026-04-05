const state = {
  data: null,
  selectedCommodity: null,
  selectedRegion: null,
  selectedModel: "Weighted Ensemble",
  selectedView: "actual-vs-forecast",
};

const elements = {
  commodity: document.getElementById("commodity-select"),
  region: document.getElementById("region-select"),
  model: document.getElementById("model-select"),
  view: document.getElementById("view-select"),
  summary: document.getElementById("series-summary"),
  metrics: document.getElementById("metrics-grid"),
  pointDetails: document.getElementById("point-details"),
  targetLabel: document.getElementById("target-label"),
  forecastHorizon: document.getElementById("forecast-horizon"),
  generatedAt: document.getElementById("generated-at"),
};

const fmtNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const fmtDate = (value) => {
  if (!value) return "N/A";
  return new Date(value).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
  });
};

const uniqueSorted = (items) => [...new Set(items)].sort((a, b) => a.localeCompare(b));

function setOptions(select, values, selectedValue) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selectedValue;
    select.appendChild(option);
  });
}

function getSeriesId() {
  const option = state.data.series_options.find(
    (item) => item.commodity_name === state.selectedCommodity && item.region === state.selectedRegion
  );
  return option ? option.series_id : null;
}

function filterHistory(seriesId) {
  return state.data.history.filter((row) => row.series_id === seriesId);
}

function filterHoldout(seriesId) {
  return state.data.holdout.filter(
    (row) => row.series_id === seriesId && row.model === state.selectedModel
  );
}

function filterFuture(seriesId) {
  return state.data.future.filter(
    (row) => row.series_id === seriesId && row.model === state.selectedModel
  );
}

function buildSeriesSummary(historyRows, futureRows) {
  const firstMonth = historyRows[0]?.month;
  const lastActualMonth = historyRows[historyRows.length - 1]?.month;
  const lastFutureMonth = futureRows[futureRows.length - 1]?.month;

  elements.summary.innerHTML = `
    <strong>${state.selectedCommodity}</strong> in <strong>${state.selectedRegion}</strong>.
    Historical window: <strong>${fmtDate(firstMonth)}</strong> to <strong>${fmtDate(lastActualMonth)}</strong>.
    Forecast window: <strong>${futureRows.length ? fmtDate(futureRows[0].month) : "N/A"}</strong>
    to <strong>${lastFutureMonth ? fmtDate(lastFutureMonth) : "N/A"}</strong>.
  `;
}

function buildMetricCards(seriesId) {
  const seriesMetric = state.data.series_metrics.find(
    (row) => row.series_id === seriesId && row.model === state.selectedModel
  );
  const globalMetric = state.data.global_metrics.find((row) => row.model === state.selectedModel);

  const cards = [
    { label: "Series RMSE", value: fmtNumber(seriesMetric?.rmse) },
    { label: "Series MAE", value: fmtNumber(seriesMetric?.mae) },
    { label: "Series R²", value: fmtNumber(seriesMetric?.r2) },
    { label: "Series Holdout Rows", value: fmtNumber(seriesMetric?.rows_evaluated, 0) },
    { label: "Global RMSE", value: fmtNumber(globalMetric?.rmse) },
    { label: "Global MAE", value: fmtNumber(globalMetric?.mae) },
    { label: "Global R²", value: fmtNumber(globalMetric?.r2) },
    { label: "Global Holdout Rows", value: fmtNumber(globalMetric?.rows_evaluated, 0) },
  ];

  elements.metrics.innerHTML = cards
    .map(
      (card) => `
        <article class="metric-card">
          <span class="metric-label">${card.label}</span>
          <strong class="metric-value">${card.value}</strong>
        </article>
      `
    )
    .join("");
}

function setPointDetails(details) {
  if (!details) {
    elements.pointDetails.className = "detail-list empty-state";
    elements.pointDetails.textContent = "Click a point to inspect it.";
    return;
  }

  const rows = [
    ["Month", fmtDate(details.month)],
    ["Series", `${state.selectedCommodity} | ${state.selectedRegion}`],
    ["Model", details.model || "Actual"],
    ["Phase", details.phase || "historical"],
    ["Actual Price", fmtNumber(details.actualPrice)],
    ["Forecast Price", fmtNumber(details.predictedPrice)],
    ["Actual YoY", fmtNumber(details.actualYoy)],
    ["Forecast YoY", fmtNumber(details.predictedYoy)],
    ["Absolute Error", fmtNumber(details.absErrorPrice)],
    ["Percent Error", fmtNumber(details.pctErrorPrice)],
  ];

  elements.pointDetails.className = "detail-list";
  elements.pointDetails.innerHTML = rows
    .map(
      ([key, value]) => `
        <div class="detail-row">
          <span class="detail-key">${key}</span>
          <strong>${value || "N/A"}</strong>
        </div>
      `
    )
    .join("");
}

function buildTrace(rows, config) {
  return {
    type: "scatter",
    mode: config.mode || "lines",
    name: config.name,
    x: rows.map((row) => row.month),
    y: rows.map((row) => row.y),
    line: config.line,
    marker: config.marker,
    customdata: rows.map((row) => ([
      row.month,
      row.actualPrice,
      row.predictedPrice,
      row.actualYoy,
      row.predictedYoy,
      row.absErrorPrice,
      row.pctErrorPrice,
      row.phase,
      row.model,
    ])),
    hovertemplate:
      "<b>%{x}</b><br>" +
      "Actual Price: %{customdata[1]:,.2f}<br>" +
      "Forecast Price: %{customdata[2]:,.2f}<br>" +
      "Actual YoY: %{customdata[3]:,.2f}%<br>" +
      "Forecast YoY: %{customdata[4]:,.2f}%<extra></extra>",
  };
}

function renderChart(seriesId) {
  const historyRows = filterHistory(seriesId);
  const holdoutRows = filterHoldout(seriesId);
  const futureRows = filterFuture(seriesId);
  buildSeriesSummary(historyRows, futureRows);
  buildMetricCards(seriesId);
  setPointDetails(null);

  const traces = [];

  if (state.selectedView !== "forecast-only") {
    traces.push(
      buildTrace(
        historyRows.map((row) => ({
          month: row.month,
          y: row.actual_price,
          actualPrice: row.actual_price,
          predictedPrice: null,
          actualYoy: row.actual_yoy,
          predictedYoy: null,
          absErrorPrice: null,
          pctErrorPrice: null,
          phase: "history",
          model: "Actual",
        })),
        {
          name: "Actual Price",
          line: { color: "#1e1b16", width: 2.4 },
          marker: { size: 5, color: "#1e1b16" },
        }
      )
    );
  }

  if (state.selectedView !== "actual-only") {
    if (holdoutRows.length) {
      traces.push(
        buildTrace(
          holdoutRows.map((row) => ({
            month: row.month,
            y: row.predicted_price,
            actualPrice: row.actual_price,
            predictedPrice: row.predicted_price,
            actualYoy: row.actual_yoy,
            predictedYoy: row.predicted_yoy,
            absErrorPrice: row.abs_error_price,
            pctErrorPrice: row.pct_error_price,
            phase: "holdout",
            model: row.model,
          })),
          {
            name: `${state.selectedModel} Holdout`,
            mode: "lines+markers",
            line: { color: "#ad3f2f", width: 2.2, dash: "dash" },
            marker: { size: 7, color: "#ad3f2f" },
          }
        )
      );
    }

    if (futureRows.length) {
      traces.push(
        buildTrace(
          futureRows.map((row) => ({
            month: row.month,
            y: row.predicted_price,
            actualPrice: null,
            predictedPrice: row.predicted_price,
            actualYoy: null,
            predictedYoy: row.predicted_yoy,
            absErrorPrice: null,
            pctErrorPrice: null,
            phase: "future",
            model: row.model,
          })),
          {
            name: `${state.selectedModel} Future Forecast`,
            mode: "lines+markers",
            line: { color: "#d77d35", width: 2.6 },
            marker: { size: 7, color: "#d77d35", symbol: "diamond" },
          }
        )
      );
    }
  }

  const splitMonth = holdoutRows[0]?.month || futureRows[0]?.month;
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.65)",
    margin: { l: 60, r: 20, t: 20, b: 60 },
    hovermode: "closest",
    legend: { orientation: "h", y: 1.08 },
    xaxis: {
      title: "Month",
      gridcolor: "rgba(30,27,22,0.08)",
    },
    yaxis: {
      title: "Price",
      gridcolor: "rgba(30,27,22,0.08)",
    },
    shapes: splitMonth ? [
      {
        type: "line",
        x0: splitMonth,
        x1: splitMonth,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "rgba(30,27,22,0.35)", dash: "dot", width: 1.5 },
      },
    ] : [],
  };

  Plotly.newPlot("chart", traces, layout, { responsive: true, displayModeBar: false });

  const chart = document.getElementById("chart");
  chart.on("plotly_click", (event) => {
    const point = event.points[0];
    const data = point.customdata;
    setPointDetails({
      month: data[0],
      actualPrice: data[1],
      predictedPrice: data[2],
      actualYoy: data[3],
      predictedYoy: data[4],
      absErrorPrice: data[5],
      pctErrorPrice: data[6],
      phase: data[7],
      model: data[8],
    });
  });
}

function refreshRegions() {
  const regions = uniqueSorted(
    state.data.series_options
      .filter((row) => row.commodity_name === state.selectedCommodity)
      .map((row) => row.region)
  );

  if (!regions.includes(state.selectedRegion)) {
    state.selectedRegion = regions[0];
  }

  setOptions(elements.region, regions, state.selectedRegion);
}

function render() {
  refreshRegions();
  const seriesId = getSeriesId();
  if (!seriesId) return;
  renderChart(seriesId);
}

async function init() {
  const response = await fetch("data/dashboard.json");
  state.data = await response.json();

  elements.targetLabel.textContent = state.data.target_label;
  elements.forecastHorizon.textContent = `${state.data.forecast_horizon_months} months`;
  elements.generatedAt.textContent = new Date(state.data.generated_at).toLocaleString();

  const commodities = uniqueSorted(state.data.series_options.map((row) => row.commodity_name));
  state.selectedCommodity = commodities[0];
  state.selectedRegion = state.data.series_options.find(
    (row) => row.commodity_name === state.selectedCommodity
  )?.region || null;

  setOptions(elements.commodity, commodities, state.selectedCommodity);
  setOptions(elements.model, state.data.models, state.selectedModel);

  elements.commodity.addEventListener("change", (event) => {
    state.selectedCommodity = event.target.value;
    render();
  });

  elements.region.addEventListener("change", (event) => {
    state.selectedRegion = event.target.value;
    render();
  });

  elements.model.addEventListener("change", (event) => {
    state.selectedModel = event.target.value;
    render();
  });

  elements.view.addEventListener("change", (event) => {
    state.selectedView = event.target.value;
    render();
  });

  render();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="layout"><section class="panel"><h2>Web app failed to load</h2><p>${error.message}</p></section></main>`;
});
