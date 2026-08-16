"use strict";

const MAX_CANDIDATES = 4;
const COLORS = ["#50d6d0", "#f3bf63", "#7eb7ff", "#ff7581", "#a98bff"];
const DASHES = ["solid", "dash", "dot", "dashdot", "longdash"];
const form = document.querySelector("#compare-form");
const baselineSelect = document.querySelector("#compare-baseline");
const candidateList = document.querySelector("#candidate-list");
const errorBox = document.querySelector("#compare-error");
const submitButton = document.querySelector("#compare-submit");
let catalog = [];

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .001)
      ? value.toExponential(3)
      : value.toLocaleString("ru-RU", { maximumFractionDigits: 6 });
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatCell(cell) {
  return cell.present ? formatValue(cell.value) : "нет поля";
}

function runCaption(run) {
  return [run.experiment, run.dataset, run.model, `seed ${run.seed ?? "—"}`, run.status]
    .filter(Boolean).join(" · ");
}

function showError(text) {
  errorBox.textContent = text;
  errorBox.classList.toggle("hidden", !text);
}

function selectedCandidates() {
  return [...candidateList.querySelectorAll("input:checked")].map(input => input.value);
}

function deactivateChart(shell) {
  shell.classList.remove("chart-interactive");
  shell.querySelector(".chart-interaction-gate").classList.remove("hidden");
}

function activateChart(shell) {
  document.querySelectorAll(".chart-shell.chart-interactive").forEach(active => {
    if (active !== shell) deactivateChart(active);
  });
  shell.classList.add("chart-interactive");
  shell.querySelector(".chart-interaction-gate").classList.add("hidden");
}

function interactiveChartShell(chart) {
  const shell = node("div", "chart-shell");
  const reset = node("button", "chart-reset", "Сбросить масштаб");
  reset.type = "button";
  reset.addEventListener("click", () => {
    Plotly.relayout(chart, { "xaxis.autorange": true, "yaxis.autorange": true });
  });
  const canvas = node("div", "chart-canvas");
  const gate = node("button", "chart-interaction-gate");
  gate.type = "button";
  gate.setAttribute("aria-label", "Включить масштабирование и панорамирование графика");
  gate.title = "Кликните, чтобы активировать график. ЛКМ — pan, колесо — zoom.";
  gate.addEventListener("click", () => activateChart(shell));
  canvas.append(chart, gate, reset);
  shell.append(canvas);
  return shell;
}

function updateCandidateState() {
  const baseline = baselineSelect.value;
  for (const input of candidateList.querySelectorAll("input")) {
    input.disabled = input.value === baseline;
    if (input.disabled) input.checked = false;
  }
  document.querySelector("#candidate-count").textContent = `${selectedCandidates().length} / ${MAX_CANDIDATES}`;
}

function renderPicker(items, requestedBaseline, requestedCandidates) {
  baselineSelect.replaceChildren(node("option", "", "Выберите baseline"));
  baselineSelect.firstElementChild.value = "";
  for (const run of items) {
    const option = document.createElement("option");
    option.value = run.run_id;
    option.textContent = `${run.run_id} — ${runCaption(run)}`;
    option.selected = run.run_id === requestedBaseline;
    baselineSelect.append(option);
  }

  candidateList.replaceChildren();
  for (const run of items) {
    const label = node("label", "candidate-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = run.run_id;
    input.checked = requestedCandidates.includes(run.run_id);
    const text = node("span");
    text.append(node("strong", "", run.run_id), node("small", "", runCaption(run)));
    label.append(input, text);
    candidateList.append(label);
  }
  updateCandidateState();
}

function comparisonParams() {
  const params = new URLSearchParams();
  params.set("baseline", baselineSelect.value);
  for (const runId of selectedCandidates()) params.append("candidate", runId);
  return params;
}

function renderLegend(payload) {
  const container = document.querySelector("#compare-legend");
  container.replaceChildren();
  const runs = [payload.baseline, ...payload.candidates];
  runs.forEach((run, index) => {
    const card = node("article", "compare-run-card");
    card.style.setProperty("--run-color", COLORS[index]);
    const heading = node("div", "compare-run-heading");
    heading.append(
      node("span", "run-color", ""),
      node("strong", "", index === 0 ? "Baseline" : `Кандидат ${index}`),
    );
    const link = node("a", "run-link", run.run_id);
    link.href = `/runs/${encodeURIComponent(run.run_id)}`;
    card.append(heading, link, node("p", "", runCaption(run)));
    container.append(card);
  });
}

function tableHeader(payload, firstColumn) {
  const row = node("tr");
  row.append(node("th", "", firstColumn), node("th", "", "Baseline"));
  payload.candidates.forEach((_, index) => row.append(node("th", "", `Кандидат ${index + 1}`)));
  return row;
}

function renderConfigDiff(payload) {
  const container = document.querySelector("#config-diff-table");
  container.replaceChildren();
  if (!payload.config_diff.length) {
    container.append(node("p", "empty", "Конфиги совпадают."));
    return;
  }
  const table = node("table", "comparison-table");
  const head = node("thead");
  head.append(tableHeader(payload, "Параметр"));
  const body = node("tbody");
  for (const item of payload.config_diff) {
    const row = node("tr");
    row.append(node("th", "comparison-key", item.key));
    row.append(node("td", "comparison-value", formatCell(item.baseline)));
    for (const cell of item.candidates) {
      row.append(node("td", cell.differs ? "comparison-value value-changed" : "comparison-value", formatCell(cell)));
    }
    body.append(row);
  }
  table.append(head, body);
  container.append(table);
}

function signed(value) {
  if (value === null || value === undefined) return null;
  return `${value > 0 ? "+" : ""}${formatValue(value)}`;
}

function renderFinalMetrics(payload) {
  const container = document.querySelector("#final-delta-table");
  container.replaceChildren();
  if (!payload.final_metrics.length) {
    container.append(node("p", "empty", "Итоговых метрик нет."));
    return;
  }
  const table = node("table", "comparison-table final-comparison-table");
  const head = node("thead");
  head.append(tableHeader(payload, "Метрика"));
  const body = node("tbody");
  for (const item of payload.final_metrics) {
    const row = node("tr");
    row.append(node("th", "comparison-key", item.key));
    row.append(node("td", "comparison-value", formatCell(item.baseline)));
    for (const cell of item.candidates) {
      const value = node("div", "metric-comparison-value", formatCell(cell));
      const deltas = [];
      const absolute = signed(cell.delta);
      if (absolute !== null) deltas.push(`Δ ${absolute}`);
      if (cell.relative_delta !== null) {
        deltas.push(`${signed(cell.relative_delta * 100)}%`);
      }
      const wrapper = node("td", "comparison-value");
      wrapper.append(value);
      if (deltas.length) wrapper.append(node("small", "metric-delta", deltas.join(" · ")));
      row.append(wrapper);
    }
    body.append(row);
  }
  table.append(head, body);
  container.append(table);
}

function renderCharts(payload) {
  const container = document.querySelector("#compare-charts");
  container.replaceChildren();
  if (!payload.metric_series.length) {
    container.append(node("p", "empty", "Числовых рядов нет."));
    return;
  }
  const allRuns = [payload.baseline, ...payload.candidates];
  const colorByRun = new Map(allRuns.map((run, index) => [run.run_id, COLORS[index]]));
  const labelByRun = new Map(allRuns.map((run, index) => [run.run_id, index === 0 ? "Baseline" : `Кандидат ${index}`]));
  const groups = new Map();
  for (const series of payload.metric_series) {
    if (!groups.has(series.group)) groups.set(series.group, []);
    groups.get(series.group).push(series);
  }

  for (const [group, seriesItems] of groups) {
    const wrapper = node("div", "compare-chart-wrap");
    const missing = [];
    const traces = [];
    const stepKeys = new Set();
    seriesItems.forEach((series, seriesIndex) => {
      if (series.missing_run_ids.length) {
        missing.push(`${series.key}: нет у ${series.missing_run_ids.map(id => labelByRun.get(id)).join(", ")}`);
      }
      for (const points of series.runs) {
        if (points.step_key) stepKeys.add(points.step_key);
        const isBaseline = points.run_id === payload.baseline.run_id;
        traces.push({
          x: points.x,
          y: points.y,
          type: "scatter",
          mode: "lines+markers",
          name: `${labelByRun.get(points.run_id)} · ${series.key}`,
          line: { color: colorByRun.get(points.run_id), width: isBaseline ? 5 : 2, dash: DASHES[seriesIndex % DASHES.length] },
          marker: { color: colorByRun.get(points.run_id), size: isBaseline ? 7 : 4, symbol: isBaseline ? "circle-open" : "circle", line: { width: isBaseline ? 2 : 0 } },
          connectgaps: false,
          hovertemplate: `${series.key}<br>${points.step_key || "x"}=%{x}<br>%{y:.5g}<extra>${labelByRun.get(points.run_id)}</extra>`,
        });
      }
    });
    const chart = node("div", "chart");
    wrapper.append(interactiveChartShell(chart));
    if (missing.length) wrapper.append(node("p", "chart-warning", missing.join("; ")));
    container.append(wrapper);
    Plotly.newPlot(chart, traces, {
      title: { text: group, font: { size: 14, color: "#dce9f0" }, x: .04 },
      paper_bgcolor: "#0a151f",
      plot_bgcolor: "#0a151f",
      font: { color: "#8da4b5", size: 10 },
      margin: { l: 54, r: 18, t: 48, b: 44 },
      xaxis: { title: stepKeys.size === 1 ? [...stepKeys][0] : "step / epoch", gridcolor: "#1d3241", zerolinecolor: "#294254" },
      yaxis: { gridcolor: "#1d3241", zerolinecolor: "#294254" },
      legend: { orientation: "h", y: 1.16, x: 1, xanchor: "right" },
      hovermode: "closest",
      dragmode: "pan",
      uirevision: `comparison:${group}`,
    }, { responsive: true, displayModeBar: false, scrollZoom: true });
  }
}

function renderWarnings(payload) {
  const panel = document.querySelector("#compare-warnings");
  const list = document.querySelector("#compare-warning-list");
  list.replaceChildren();
  panel.classList.toggle("hidden", !payload.warnings.length);
  for (const warning of payload.warnings) list.append(node("li", "", warning.message));
}

function renderComparison(payload) {
  renderLegend(payload);
  renderWarnings(payload);
  renderConfigDiff(payload);
  renderFinalMetrics(payload);
  renderCharts(payload);
  document.querySelector("#compare-results").classList.remove("hidden");
}

async function loadComparison() {
  const candidates = selectedCandidates();
  if (!baselineSelect.value) {
    showError("Выберите baseline.");
    return;
  }
  if (!candidates.length) {
    showError("Выберите хотя бы один кандидат.");
    return;
  }
  if (candidates.length > MAX_CANDIDATES) {
    showError(`Можно выбрать не больше ${MAX_CANDIDATES} кандидатов.`);
    return;
  }
  const params = comparisonParams();
  submitButton.disabled = true;
  showError("");
  try {
    const response = await fetch(`/api/compare?${params}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    history.replaceState(null, "", `/compare?${params}`);
    renderComparison(payload);
  } catch (error) {
    showError(`Не удалось сравнить запуски: ${error.message}`);
  } finally {
    submitButton.disabled = false;
  }
}

async function loadCatalog() {
  const requested = new URLSearchParams(location.search);
  const requestedBaseline = requested.get("baseline") || "";
  const requestedCandidates = requested.getAll("candidate").slice(0, MAX_CANDIDATES);
  try {
    const response = await fetch("/api/runs?limit=500", { cache: "no-store" });
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    const payload = await response.json();
    catalog = payload.items;
    renderPicker(catalog, requestedBaseline, requestedCandidates);
    if (requestedBaseline && requestedCandidates.length) loadComparison();
  } catch (error) {
    showError(`Не удалось загрузить каталог: ${error.message}`);
  }
}

baselineSelect.addEventListener("change", updateCandidateState);
candidateList.addEventListener("change", event => {
  if (selectedCandidates().length > MAX_CANDIDATES) {
    event.target.checked = false;
    showError(`Можно выбрать не больше ${MAX_CANDIDATES} кандидатов.`);
  } else {
    showError("");
  }
  updateCandidateState();
});
form.addEventListener("submit", event => { event.preventDefault(); loadComparison(); });
form.addEventListener("reset", () => setTimeout(() => {
  document.querySelector("#compare-results").classList.add("hidden");
  document.querySelector("#compare-warnings").classList.add("hidden");
  history.replaceState(null, "", "/compare");
  showError("");
  updateCandidateState();
}, 0));

loadCatalog();

document.addEventListener("pointerdown", event => {
  const active = document.querySelector(".chart-shell.chart-interactive");
  if (active && !active.contains(event.target)) deactivateChart(active);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    document.querySelectorAll(".chart-shell.chart-interactive").forEach(deactivateChart);
  }
});
