"use strict";

const runId = JSON.parse(document.querySelector("#run-id-data").textContent);
const encodedRunId = encodeURIComponent(runId);
const statusNames = {
  queued: "в очереди", running: "в работе", paused: "пауза",
  completed: "завершён", failed: "ошибка", cancelled: "отменён",
};
const terminalStatuses = new Set(["completed", "failed", "cancelled"]);
let detail = null;
let detailSignature = "";
let logOffset = 0;
let logLoading = false;
let logMissing = false;
let rerunPreview = null;
const rerunInputs = new Map();
let modelSignature = "";
let modelLoadToken = 0;
let selectedLayerId = null;
const selectedTensorByLayer = new Map();

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
      ? value.toExponential(3) : value.toLocaleString("ru-RU", { maximumFractionDigits: 6 });
  }
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium", timeStyle: "medium",
  }).format(date);
}

function formatBytes(value) {
  if (value < 1024) return `${value} Б`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} КиБ`;
  return `${(value / 1024 ** 2).toFixed(1)} МиБ`;
}

function renderKeyValues(selector, values) {
  const container = document.querySelector(selector);
  container.replaceChildren();
  const entries = Object.entries(values || {});
  if (!entries.length) {
    container.append(node("p", "empty", "Нет данных"));
    return;
  }
  for (const [key, value] of entries) {
    const row = node("div", "kv-row");
    row.append(node("div", "kv-key", key), node("div", "kv-value", formatValue(value)));
    container.append(row);
  }
}

function showNotice(selector, text) {
  const notice = document.querySelector(selector);
  notice.textContent = text;
  notice.classList.toggle("hidden", !text);
}

function renderHeader(manifest, config, metrics) {
  document.querySelector("#run-title").textContent = manifest.run_id;
  document.querySelector("#run-experiment").textContent = manifest.experiment || "Запуск";
  const status = document.querySelector("#run-status");
  status.className = `status status-${manifest.status}`;
  status.textContent = statusNames[manifest.status] || manifest.status;
  const meta = [
    config.dataset, config.model || config.name,
    `seed ${config.seed ?? "—"}`, formatDate(manifest.started_at),
    manifest.duration_sec == null ? null : `${formatValue(manifest.duration_sec)} с`,
    manifest.adapted_from_legacy ? "legacy contract" : "contract v1",
  ].filter(Boolean);
  document.querySelector("#run-meta").textContent = meta.join(" · ");

  const git = metrics.git || {};
  const dirtyFiles = Array.isArray(git.dirty_files) ? git.dirty_files : [];
  showNotice("#dirty-notice", git.dirty
    ? `DIRTY RUN — результат нельзя точно воспроизвести.\n${dirtyFiles.join("\n")}` : "");
  showNotice("#failed-notice", manifest.status === "failed"
    ? `Прогон завершился ошибкой.\n${metrics.error || "Трассировка отсутствует."}` : "");
  const parentNotice = document.querySelector("#parent-notice");
  parentNotice.replaceChildren();
  parentNotice.classList.toggle("hidden", !manifest.parent_run_id);
  if (manifest.parent_run_id) {
    parentNotice.append("Повторный запуск от ");
    const parentLink = node("a", "run-link", manifest.parent_run_id);
    parentLink.href = `/runs/${encodeURIComponent(manifest.parent_run_id)}`;
    parentNotice.append(parentLink);
  }
  const compareLink = document.querySelector("#compare-run-link");
  const compareParams = new URLSearchParams();
  if (manifest.parent_run_id) {
    compareParams.set("baseline", manifest.parent_run_id);
    compareParams.append("candidate", manifest.run_id);
    compareLink.textContent = "Сравнить с родителем";
  } else {
    compareParams.set("baseline", manifest.run_id);
    compareLink.textContent = "Добавить к сравнению";
  }
  compareLink.href = `/compare?${compareParams}`;
}

function renderFinal(values) {
  const container = document.querySelector("#final-metrics");
  container.replaceChildren();
  const entries = Object.entries(values || {});
  if (!entries.length) {
    container.append(node("p", "empty", "Итоговые метрики ещё не записаны."));
    return;
  }
  for (const [key, value] of entries) {
    const card = node("article", "stat");
    card.append(node("span", "", key), node("strong", "", formatValue(value)));
    container.append(card);
  }
}

function metricGroups(rows) {
  if (!rows.length) return { stepKey: "step", groups: new Map() };
  const stepKey = ["step", "epoch"].find(key => key in rows[0]) || Object.keys(rows[0])[0];
  const groups = new Map();
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (key === stepKey || typeof value !== "number" || !Number.isFinite(value)) continue;
      const group = key.includes("/") ? key.split("/", 1)[0] : "scalar";
      if (!groups.has(group)) groups.set(group, new Map());
      if (!groups.get(group).has(key)) groups.get(group).set(key, { x: [], y: [] });
      groups.get(group).get(key).x.push(row[stepKey]);
      groups.get(group).get(key).y.push(value);
    }
  }
  return { stepKey, groups };
}

function renderCharts(metrics) {
  const rows = Array.isArray(metrics.epochs) ? metrics.epochs : [];
  const signature = JSON.stringify(rows);
  if (signature === detailSignature) return;
  detailSignature = signature;
  const charts = document.querySelector("#charts");
  charts.replaceChildren();
  const { stepKey, groups } = metricGroups(rows);
  if (!groups.size) {
    charts.append(node("p", "empty", "Скалярных рядов пока нет."));
    return;
  }
  for (const [group, series] of groups) {
    const chart = node("div", "chart");
    chart.setAttribute("aria-label", `График ${group}`);
    charts.append(interactiveChartShell(chart));
    const traces = [...series.entries()].map(([key, points]) => ({
      x: points.x, y: points.y, type: "scatter", mode: "lines+markers",
      name: key.includes("/") ? key.split("/").slice(1).join("/") : key,
      line: { width: 2 }, marker: { size: 4 }, hovertemplate: `${key}<br>${stepKey}=%{x}<br>%{y:.5g}<extra></extra>`,
    }));
    const allPositive = traces.every(trace => trace.y.every(value => value > 0));
    const useLog = ["w_norm", "grad_norm", "act_rms", "act_max"].includes(group) && allPositive;
    Plotly.newPlot(chart, traces, {
      title: { text: group, font: { size: 14, color: "#dce9f0" }, x: .04 },
      paper_bgcolor: "#0a151f", plot_bgcolor: "#0a151f", font: { color: "#8da4b5", size: 10 },
      margin: { l: 54, r: 18, t: 46, b: 44 },
      xaxis: { title: stepKey, gridcolor: "#1d3241", zerolinecolor: "#294254" },
      yaxis: { type: useLog ? "log" : "linear", gridcolor: "#1d3241", zerolinecolor: "#294254" },
      legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
      hovermode: "x unified", dragmode: "pan", uirevision: `${runId}:${group}`,
    }, { responsive: true, displayModeBar: false, scrollZoom: true });
  }
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

function formatShape(shape) {
  if (!Array.isArray(shape)) return "—";
  if (!shape.length) return "scalar";
  return shape.map(value => value == null ? "?" : value).join(" × ");
}

function validateModelPayload(payload) {
  if (!payload || payload.schema_version !== 1 || payload.kind !== "model") {
    throw new Error("Поддерживается только model.json версии 1.");
  }
  if (payload.run_id !== runId) {
    throw new Error(`model.json принадлежит другому запуску: ${payload.run_id || "—"}`);
  }
  if (!Array.isArray(payload.layers) || !payload.layers.length) {
    throw new Error("model.json не содержит слоёв.");
  }
  if (payload.capture_batch_size != null &&
      (!Number.isInteger(payload.capture_batch_size) || payload.capture_batch_size < 1)) {
    throw new Error("capture_batch_size должен быть положительным целым числом.");
  }
  const layerIds = new Set();
  for (const layer of payload.layers) {
    if (!layer || typeof layer.id !== "string" || !layer.id || layerIds.has(layer.id)) {
      throw new Error("model.json содержит пустой или повторяющийся id слоя.");
    }
    layerIds.add(layer.id);
    if (!Array.isArray(layer.tensors)) throw new Error(`У слоя ${layer.id} нет списка tensors.`);
  }
  if (!Array.isArray(payload.connections)) throw new Error("В model.json нет списка connections.");
  for (const connection of payload.connections) {
    if (!layerIds.has(connection.source) || !layerIds.has(connection.target)) {
      throw new Error(`Связь ${connection.source || "?"} → ${connection.target || "?"} ссылается на неизвестный слой.`);
    }
  }
}

function setModelState(message, kind = "empty") {
  const state = document.querySelector("#model-state");
  state.replaceChildren();
  const content = node("p", kind === "error" ? "notice notice-error model-notice" : "empty", message);
  state.append(content);
  state.classList.remove("hidden");
  document.querySelector("#model-inspector").classList.add("hidden");
}

function modelNode(layer) {
  const weight = layer.tensors.find(tensor => tensor.role === "parameter" && tensor.name === "weight");
  const button = node("button", "model-node");
  button.type = "button";
  button.dataset.layerId = layer.id;
  button.setAttribute("aria-label", `Открыть слой ${layer.id}`);
  button.append(
    node("span", "model-node-id", layer.id),
    node("strong", "model-node-type", layer.type || "unknown"),
    node("span", "model-node-shape", weight ? `weight ${formatShape(weight.shape)}` : "weight не сохранён"),
    node("span", "model-node-params", `${formatValue(layer.parameter_count || 0)} параметров`),
  );
  button.addEventListener("click", () => {
    selectedLayerId = layer.id;
    renderSelectedLayer(currentModel);
  });
  return button;
}

let currentModel = null;

function renderModelFacts(model, config) {
  const facts = document.querySelector("#model-facts");
  facts.replaceChildren();
  const configuredBatch = config?.batch_size ?? config?.batch;
  if (configuredBatch != null) facts.append(fact("Батч обучения", formatValue(configuredBatch)));
  facts.append(
    fact("Батч снимка", model.capture_batch_size == null ? "не записан" : formatValue(model.capture_batch_size)),
    fact("Слои", formatValue(model.layers.length)),
    fact("Параметры", formatValue(model.layers.reduce((sum, layer) => sum + (layer.parameter_count || 0), 0))),
  );
}

function renderModelGraph(model) {
  const graph = document.querySelector("#model-graph");
  graph.replaceChildren();
  model.layers.forEach((layer, index) => {
    if (index) {
      const previous = model.layers[index - 1];
      const connection = model.connections.find(item =>
        item.source === previous.id && item.target === layer.id);
      const edge = node("div", connection ? `model-edge model-edge-${connection.kind}` : "model-edge model-edge-order");
      edge.append(
        node("span", "model-edge-arrow", connection?.kind === "learning" ? "⇢" : connection ? "→" : "⋯"),
        node("small", "", connection?.kind || "порядок"),
      );
      graph.append(edge);
    }
    graph.append(modelNode(layer));
  });

  const connections = document.querySelector("#model-connections");
  connections.replaceChildren();
  if (!model.connections.length) {
    connections.append(node("span", "model-connection-empty", "Связи не сохранены"));
    return;
  }
  for (const connection of model.connections) {
    connections.append(node(
      "span",
      `model-connection model-connection-${connection.kind}`,
      `${connection.source} → ${connection.target} · ${connection.kind}`,
    ));
  }
}

function fact(label, value) {
  const item = node("div", "layer-fact");
  item.append(node("dt", "", label), node("dd", "", value));
  return item;
}

function summaryValue(key, value) {
  if (value == null) return "—";
  if (key === "sparsity") return `${formatValue(value * 100)} %`;
  return formatValue(value);
}

function tensorRows(values, shape) {
  if (!shape.length) return [{ label: "value", values: [values] }];
  if (shape.length === 1) return [{ label: "value", values }];
  const rows = [];
  function visit(value, path, depth) {
    if (depth === shape.length - 1) {
      rows.push({ label: `[${path.join(", ")}]`, values: value });
      return;
    }
    value.forEach((child, index) => visit(child, [...path, index], depth + 1));
  }
  visit(values, [], 0);
  return rows;
}

function renderTensorTable(tensor, rows) {
  const wrap = node("div", "tensor-values-scroll");
  const table = node("table", "tensor-values-table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(node("th", "tensor-index", "индекс"));
  const columnCount = Math.max(0, ...rows.map(row => row.values.length));
  for (let index = 0; index < columnCount; index += 1) headRow.append(node("th", "", `[${index}]`));
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const tableRow = document.createElement("tr");
    tableRow.append(node("th", "tensor-index", row.label));
    row.values.forEach(value => tableRow.append(node("td", "", formatValue(value))));
    body.append(tableRow);
  }
  table.append(head, body);
  wrap.append(table);
  wrap.setAttribute("aria-label", `Числовые значения тензора ${tensor.name}`);
  return wrap;
}

function heatColor(value, maximum) {
  if (!maximum || value === 0) return "#132635";
  const strength = .14 + .78 * Math.min(1, Math.abs(value) / maximum);
  return value < 0 ? `rgba(255, 117, 129, ${strength})` : `rgba(80, 214, 208, ${strength})`;
}

function renderTensorHeatmap(tensor, rows) {
  const values = rows.flatMap(row => row.values).filter(value => typeof value === "number" && Number.isFinite(value));
  const maximum = Math.max(0, ...values.map(Math.abs));
  const columnCount = Math.max(0, ...rows.map(row => row.values.length));
  const section = node("section", "tensor-heatmap-section");
  const heading = node("div", "tensor-view-heading");
  heading.append(node("h4", "", "Heatmap"));
  const scale = node("span", "tensor-heatmap-scale", `−${formatValue(maximum)} · 0 · +${formatValue(maximum)}`);
  heading.append(scale);
  const scroll = node("div", "tensor-heatmap-scroll");
  const grid = node("div", "tensor-heatmap");
  grid.style.setProperty("--tensor-columns", String(columnCount));
  grid.style.minWidth = `${Math.max(240, 70 + columnCount * 38)}px`;
  grid.append(node("span", "tensor-heatmap-corner", ""));
  for (let index = 0; index < columnCount; index += 1) grid.append(node("span", "tensor-heatmap-axis", String(index)));
  rows.forEach((row, rowIndex) => {
    grid.append(node("span", "tensor-heatmap-axis tensor-heatmap-row", row.label));
    row.values.forEach((value, columnIndex) => {
      const cell = node("span", "tensor-heatmap-cell", formatValue(value));
      cell.style.background = heatColor(value, maximum);
      cell.title = `${tensor.name}${row.label === "value" ? "" : row.label}[${columnIndex}] = ${formatValue(value)}`;
      grid.append(cell);
    });
  });
  scroll.append(grid);
  section.append(heading, scroll);
  return section;
}

const omittedReasons = {
  size_limit: "Полные значения не включены: тензор превышает лимит инспектора.",
  non_finite: "Полные значения не включены: тензор содержит NaN или Infinity.",
  empty_tensor: "Тензор пуст.",
  unsupported_json_dtype: "Полные значения не включены для этого dtype.",
};

function renderTensor(layer, tensor, container) {
  container.replaceChildren();
  const heading = node("div", "tensor-heading");
  const title = node("div", "");
  title.append(node("p", "eyebrow", tensor.role || "tensor"), node("h4", "", tensor.name));
  const badges = node("div", "tensor-badges");
  badges.append(
    node("span", "tensor-badge", tensor.dtype || "unknown"),
    node("span", "tensor-badge", formatShape(tensor.shape)),
    node("span", `tensor-badge tensor-mode-${tensor.value_mode}`, tensor.value_mode),
  );
  heading.append(title, badges);
  container.append(heading);

  const summary = tensor.summary;
  if (summary) {
    const labels = {
      element_count: "элементов", finite_count: "конечных", non_finite_count: "NaN / Inf",
      min: "min", max: "max", mean: "mean", std: "std",
      l1_norm: "L1", l2_norm: "L2", sparsity: "sparsity",
    };
    const summaryGrid = node("dl", "tensor-summary");
    for (const [key, label] of Object.entries(labels)) {
      const item = node("div", "tensor-summary-item");
      item.append(node("dt", "", label), node("dd", "", summaryValue(key, summary[key])));
      summaryGrid.append(item);
    }
    container.append(summaryGrid);
  } else {
    container.append(node("p", "model-inline-empty", "Для тензора сохранены только метаданные."));
  }

  if (tensor.value_mode !== "full" || tensor.values == null) {
    if (tensor.values_omitted_reason) {
      container.append(node("p", "model-inline-empty", omittedReasons[tensor.values_omitted_reason] || `Значения не включены: ${tensor.values_omitted_reason}.`));
    }
    return;
  }

  const rows = tensorRows(tensor.values, tensor.shape || []);
  const valuesHeading = node("div", "tensor-view-heading");
  valuesHeading.append(node("h4", "", "Числа"), node("span", "", `${formatValue(summary?.element_count || 0)} значений`));
  container.append(valuesHeading, renderTensorTable(tensor, rows), renderTensorHeatmap(tensor, rows));
}

function renderSelectedLayer(model) {
  const layer = model.layers.find(item => item.id === selectedLayerId) || model.layers[0];
  selectedLayerId = layer.id;
  document.querySelectorAll(".model-node").forEach(item => {
    const selected = item.dataset.layerId === layer.id;
    item.classList.toggle("model-node-selected", selected);
    item.setAttribute("aria-pressed", String(selected));
  });

  const inspector = document.querySelector("#layer-inspector");
  inspector.replaceChildren();
  const header = node("div", "layer-heading");
  const title = node("div", "");
  title.append(node("p", "eyebrow", "Выбранный слой"), node("h3", "", layer.id));
  const badges = node("div", "layer-badges");
  badges.append(node("span", "layer-badge", layer.type || "unknown"));
  if (layer.activation) badges.append(node("span", "layer-badge layer-badge-accent", layer.activation));
  header.append(title, badges);
  const facts = node("dl", "layer-facts");
  const weight = layer.tensors.find(tensor => tensor.role === "parameter" && tensor.name === "weight");
  const bias = layer.tensors.find(tensor => tensor.role === "parameter" && tensor.name === "bias");
  facts.append(
    fact("Матрица weight", weight ? formatShape(weight.shape) : "—"),
    fact("Вектор bias", bias ? formatShape(bias.shape) : "—"),
    fact("Активация", layer.activation || "—"),
    fact("Параметры", formatValue(layer.parameter_count || 0)),
  );
  inspector.append(header, facts);

  if (!layer.tensors.length) {
    inspector.append(node("p", "model-inline-empty", "У слоя нет сохранённых тензоров."));
    return;
  }
  let tensorName = selectedTensorByLayer.get(layer.id);
  if (!layer.tensors.some(item => item.name === tensorName)) tensorName = layer.tensors[0].name;
  selectedTensorByLayer.set(layer.id, tensorName);
  const tabs = node("div", "tensor-tabs");
  const tensorBody = node("div", "tensor-body");
  for (const tensor of layer.tensors) {
    const tab = node("button", tensor.name === tensorName ? "tensor-tab tensor-tab-selected" : "tensor-tab", tensor.name);
    tab.type = "button";
    tab.addEventListener("click", () => {
      selectedTensorByLayer.set(layer.id, tensor.name);
      renderSelectedLayer(model);
    });
    tabs.append(tab);
  }
  inspector.append(tabs, tensorBody);
  renderTensor(layer, layer.tensors.find(item => item.name === tensorName), tensorBody);
}

function renderModel(model, config) {
  currentModel = model;
  if (!model.layers.some(layer => layer.id === selectedLayerId)) selectedLayerId = model.layers[0].id;
  const capture = [
    model.model_name,
    `${model.layers.length} слоёв`,
    model.capture_phase ? `фаза ${model.capture_phase}` : null,
    model.step == null ? null : `шаг ${formatValue(model.step)}`,
    model.captured_at ? formatDate(model.captured_at) : null,
  ].filter(Boolean);
  document.querySelector("#model-caption").textContent = capture.join(" · ");
  document.querySelector("#model-state").classList.add("hidden");
  document.querySelector("#model-inspector").classList.remove("hidden");
  renderModelFacts(model, config);
  renderModelGraph(model);
  renderSelectedLayer(model);
}

async function loadModel(payload) {
  const modelPath = payload.manifest.artifacts?.model || "model.json";
  const artifact = (payload.artifacts || []).find(item => item.path === modelPath);
  if (!artifact) {
    modelSignature = "";
    modelLoadToken += 1;
    currentModel = null;
    selectedLayerId = null;
    document.querySelector("#model-caption").textContent = "Инспекционный снимок не найден";
    const active = !terminalStatuses.has(payload.manifest.status);
    setModelState(active
      ? "model.json ещё не создан. Снимок появится после безопасной точки экспорта."
      : "Для этого запуска model.json не сохранён. Это нормальное состояние для старых прогонов.");
    return;
  }
  const signature = `${modelPath}:${artifact.size_bytes}:${artifact.modified_at}`;
  if (signature === modelSignature) return;
  modelSignature = signature;
  const token = ++modelLoadToken;
  document.querySelector("#model-caption").textContent = "Читаем инспекционный снимок…";
  setModelState("Загрузка model.json…");
  const encodedPath = modelPath.split("/").map(encodeURIComponent).join("/");
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/artifacts/${encodedPath}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    const model = await response.json();
    validateModelPayload(model);
    if (token === modelLoadToken) renderModel(model, payload.config);
  } catch (error) {
    if (token !== modelLoadToken) return;
    document.querySelector("#model-caption").textContent = "model.json не удалось прочитать";
    setModelState(`Некорректный инспекционный снимок: ${error.message}`, "error");
  }
}

function renderArtifacts(items) {
  const container = document.querySelector("#artifact-list");
  container.replaceChildren();
  document.querySelector("#artifact-count").textContent = `${items.length} файлов`;
  for (const item of items) {
    const row = node("div", "artifact-row");
    const link = node("a", "artifact-path", item.path);
    link.href = `/api/runs/${encodedRunId}/artifacts/${item.path.split("/").map(encodeURIComponent).join("/")}`;
    link.target = "_blank";
    const time = node("time", "artifact-meta", formatDate(item.modified_at));
    time.dateTime = item.modified_at;
    row.append(link, node("span", "artifact-meta", formatBytes(item.size_bytes)), time);
    container.append(row);
  }
}

function renderDetail(payload) {
  detail = payload;
  renderHeader(payload.manifest, payload.config, payload.metrics);
  renderFinal(payload.metrics.final || {});
  renderKeyValues("#config-table", payload.config);
  renderKeyValues("#env-table", payload.metrics.env || {});
  renderKeyValues("#git-table", payload.metrics.git || {});
  renderCharts(payload.metrics);
  renderArtifacts(payload.artifacts || []);
  loadModel(payload);
  const logPath = payload.manifest.artifacts.log || "run.log";
  const encodedLogPath = logPath.split("/").map(encodeURIComponent).join("/");
  document.querySelector("#download-log").href = `/api/runs/${encodedRunId}/artifacts/${encodedLogPath}`;
}

async function fetchDetail() {
  try {
    const response = await fetch(`/api/runs/${encodedRunId}`, { cache: "no-store" });
    if (!response.ok) throw new Error((await response.json()).detail || `API ответил ${response.status}`);
    renderDetail(await response.json());
    showNotice("#page-error", "");
  } catch (error) {
    showNotice("#page-error", `Не удалось прочитать прогон: ${error.message}`);
  }
}

async function pollLog(reset = false) {
  if (logLoading) return;
  logLoading = true;
  const output = document.querySelector("#run-log");
  if (reset) { logOffset = 0; logMissing = false; output.textContent = ""; }
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/log?offset=${logOffset}`, { cache: "no-store" });
    if (response.status === 404) {
      output.textContent = "run.log ещё не создан.";
      logMissing = true;
      return;
    }
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    const payload = await response.json();
    if (logMissing) { output.textContent = ""; logMissing = false; }
    if (payload.text) output.textContent += payload.text;
    logOffset = payload.next_offset;
    document.querySelector("#log-progress").textContent = `${formatBytes(logOffset)} / ${formatBytes(payload.size_bytes)}`;
    if (document.querySelector("#scroll-log").checked) output.scrollTop = output.scrollHeight;
  } catch (error) {
    document.querySelector("#log-progress").textContent = `Ошибка: ${error.message}`;
  } finally {
    logLoading = false;
  }
}

function rerunControl(field, index) {
  const id = `rerun-field-${index}`;
  const value = field.value;
  let input;
  if (typeof value === "boolean") {
    input = document.createElement("select");
    for (const optionValue of [true, false]) {
      const option = document.createElement("option");
      option.value = String(optionValue);
      option.textContent = String(optionValue);
      option.selected = value === optionValue;
      input.append(option);
    }
  } else if (typeof value === "number") {
    input = document.createElement("input");
    input.type = "number";
    input.step = Number.isInteger(value) ? "1" : "any";
    input.value = String(value);
  } else if (Array.isArray(value)) {
    input = document.createElement("textarea");
    input.rows = 2;
    input.value = JSON.stringify(value);
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = value === null ? "null" : String(value);
  }
  input.id = id;
  input.dataset.rerunKey = field.key;
  input.addEventListener("input", renderRerunDiff);
  input.addEventListener("change", renderRerunDiff);
  rerunInputs.set(field.key, { input, original: value });

  const row = node("div", "rerun-field");
  const label = document.createElement("label");
  label.htmlFor = id;
  label.append(node("strong", "", field.key), node("span", "", Array.isArray(value) ? "array" : typeof value));
  row.append(label, input);
  return row;
}

function parseRerunValue(input, original) {
  if (typeof original === "boolean") return input.value === "true";
  if (typeof original === "number") {
    const value = Number(input.value);
    if (!Number.isFinite(value) || (Number.isInteger(original) && !Number.isInteger(value))) {
      throw new Error(`${input.dataset.rerunKey}: ожидалось ${Number.isInteger(original) ? "целое" : "число"}`);
    }
    return value;
  }
  if (Array.isArray(original)) {
    const value = JSON.parse(input.value);
    if (!Array.isArray(value)) throw new Error(`${input.dataset.rerunKey}: ожидался JSON-массив`);
    return value;
  }
  if (original === null) return JSON.parse(input.value);
  return input.value;
}

function collectRerunConfig() {
  const config = JSON.parse(JSON.stringify(rerunPreview.config));
  for (const [key, state] of rerunInputs) {
    config[key] = parseRerunValue(state.input, state.original);
  }
  return config;
}

function renderRerunDiff() {
  if (!rerunPreview) return;
  const container = document.querySelector("#rerun-diff");
  container.replaceChildren();
  try {
    const config = collectRerunConfig();
    const changed = Object.keys(config).filter(key =>
      JSON.stringify(config[key]) !== JSON.stringify(rerunPreview.config[key]));
    if (!changed.length) {
      container.append(node("p", "empty", "Параметры не изменены — можно повторить конфиг без правок."));
    } else {
      for (const key of changed) {
        const row = node("div", "rerun-diff-row");
        row.append(
          node("strong", "", key),
          node("code", "rerun-before", formatValue(rerunPreview.config[key])),
          node("span", "rerun-arrow", "→"),
          node("code", "rerun-after", formatValue(config[key])),
        );
        container.append(row);
      }
    }
    showNotice("#rerun-error", "");
  } catch (error) {
    container.append(node("p", "empty", "Исправьте значение, чтобы увидеть diff."));
    showNotice("#rerun-error", error.message);
  }
}

function renderRerunForm(preview) {
  rerunPreview = preview;
  rerunInputs.clear();
  const fields = document.querySelector("#rerun-fields");
  fields.replaceChildren();
  const editable = preview.fields.filter(field => field.editable);
  editable.forEach((field, index) => fields.append(rerunControl(field, index)));
  const locked = preview.fields.filter(field => !field.editable).map(field => field.key);
  showNotice("#rerun-message",
    `${editable.length} параметров можно менять. Зафиксированы: ${locked.join(", ") || "нет"}.`);
  document.querySelector("#rerun-form").classList.remove("hidden");
  renderRerunDiff();
}

async function loadRerunPreview() {
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/rerun`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    renderRerunForm(payload);
  } catch (error) {
    showNotice("#rerun-message", error.message);
  }
}

document.querySelector("#rerun-reset").addEventListener("click", () => {
  if (rerunPreview) renderRerunForm(rerunPreview);
});
document.querySelector("#rerun-form").addEventListener("submit", async event => {
  event.preventDefault();
  const submit = document.querySelector("#rerun-submit");
  try {
    const config = collectRerunConfig();
    submit.disabled = true;
    submit.textContent = "Ставим в очередь…";
    const response = await fetch(`/api/runs/${encodedRunId}/rerun`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    window.location.assign(payload.location);
  } catch (error) {
    showNotice("#rerun-error", error.message);
    submit.disabled = false;
    submit.textContent = "Поставить в очередь";
  }
});

document.querySelector("#reload-log").addEventListener("click", () => pollLog(true));
document.addEventListener("pointerdown", event => {
  const active = document.querySelector(".chart-shell.chart-interactive");
  if (active && !active.contains(event.target)) deactivateChart(active);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    document.querySelectorAll(".chart-shell.chart-interactive").forEach(deactivateChart);
  }
});
setInterval(() => {
  if (document.querySelector("#follow-log").checked) pollLog();
}, 1500);
setInterval(() => {
  if (!detail || !terminalStatuses.has(detail.manifest.status)) fetchDetail();
}, 5000);
fetchDetail();
pollLog(true);
loadRerunPreview();
