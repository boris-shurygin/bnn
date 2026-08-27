"use strict";

const runId = JSON.parse(document.querySelector("#run-id-data").textContent);
const encodedRunId = encodeURIComponent(runId);
const statusNames = {
  queued: "в очереди", running: "в работе", paused: "пауза",
  suspended: "гибернация", interrupted: "прерван",
  completed: "завершён", failed: "ошибка", cancelled: "отменён",
};
const terminalStatuses = new Set(["completed", "failed", "cancelled"]);
let detail = null;
let detailSignature = "";
let logOffset = 0;
let logLoading = false;
let logMissing = false;
let controlPollLoading = false;
let controlCommandLoading = false;
let controlRevision = 0;
let currentControl = null;
let rerunPreview = null;
const rerunInputs = new Map();
let modelSignature = "";
let modelLoadToken = 0;
let selectedLayerId = null;
const selectedTensorByLayer = new Map();
let neuronVisualizationSignature = "";
let neuronVisualizationLoadToken = 0;
let neuronVisualizations = null;
let neuronVisualizationError = "";
const selectedNeuronByLayer = new Map();
let xorEventSeq = 0;
let xorSnapshot = null;
let xorSnapshotLoading = false;
let xorCanSetInput = false;
let xorInputLoading = false;
let modelDebugEventSeq = 0;
let modelDebugSnapshot = null;
let modelDebugSnapshotLoading = false;
let modelDebugCanSetInput = false;
let modelDebugInputLoading = false;
let xorTrainingEventSeq = 0;
let xorTrainingEvents = [];
let xorTrainingIndex = -1;
let xorTrainingLoading = false;
let xorTrainingPlaying = false;
let xorTrainingTimer = null;
const xorTrainingSnapshots = new Map();
let lastUserInteractionAt = Date.now();

function debugCapabilities(config) {
  if (config?.debug && typeof config.debug === "object" && !Array.isArray(config.debug)) {
    return config.debug;
  }
  // Совместимость с debug-сессиями, созданными до появления явных capabilities.
  return config?.experiment === "xor_interactive"
    ? {
      protocol: "model_debug_v1", renderer: "xor_neurons_v1",
      accepts_input: true, input_size: 2, supports_step: true, step_scope: "layer",
    }
    : null;
}

function isDebugSession(config) {
  return debugCapabilities(config)?.protocol === "model_debug_v1";
}

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function statusBadge(status) {
  return node("span", `status status-${status}`, statusNames[status] || status);
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
  document.querySelector("#delete-run").classList.toggle(
    "hidden", !terminalStatuses.has(manifest.status),
  );
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
    parentNotice.append(isDebugSession(config)
      ? "Отладочная сессия модели из " : "Повторный запуск от ");
    const parentLink = node("a", "run-link", manifest.parent_run_id);
    parentLink.href = `/runs/${encodeURIComponent(manifest.parent_run_id)}`;
    parentNotice.append(parentLink);
  }
  const compareLink = document.querySelector("#compare-run-link");
  const compareParams = new URLSearchParams();
  if (manifest.parent_run_id) {
    compareParams.set("baseline", manifest.parent_run_id);
    compareParams.append("candidate", manifest.run_id);
    compareLink.textContent = isDebugSession(config)
      ? "Сравнить с исходным" : "Сравнить с родителем";
  } else {
    compareParams.set("baseline", manifest.run_id);
    compareLink.textContent = "Добавить к сравнению";
  }
  compareLink.href = `/compare?${compareParams}`;
}

function renderDebugSessions(items) {
  const panel = document.querySelector("#debug-sessions-panel");
  const list = document.querySelector("#debug-sessions-list");
  list.replaceChildren();
  panel.classList.toggle("hidden", !items.length);
  document.querySelector("#debug-sessions-count").textContent = items.length
    ? `${items.length} шт.` : "";
  for (const run of items) {
    const item = node("li");
    const identity = node("div");
    const link = node("a", "run-link", run.run_id);
    link.href = `/runs/${encodeURIComponent(run.run_id)}`;
    identity.append(link, node("span", "debug-badge", "debug"));
    identity.append(node("div", "cell-sub", [run.experiment, run.model].filter(Boolean).join(" · ")));
    const meta = node("div", "child-run-meta");
    meta.append(statusBadge(run.status), node("span", "", formatDate(run.started_at)));
    item.append(identity, meta);
    list.append(item);
  }
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

function validateNeuronVisualizations(payload) {
  if (!payload || payload.schema_version !== 1 || payload.kind !== "neuron_visualizations") {
    throw new Error("Поддерживается только neuron_visualizations версии 1.");
  }
  if (payload.run_id !== runId) {
    throw new Error(`Neuron visualizations принадлежит другому запуску: ${payload.run_id || "—"}`);
  }
  if (!Array.isArray(payload.input_shape) || payload.input_shape.length !== 2
      || payload.input_shape.some(value => !Number.isInteger(value) || value < 1)) {
    throw new Error("Neuron visualizations требует двумерный input_shape.");
  }
  if (!Array.isArray(payload.layers)) throw new Error("Neuron visualizations не содержит layers.");
  const layerIds = new Set();
  for (const layer of payload.layers) {
    if (!layer || typeof layer.layer_id !== "string" || !layer.layer_id
        || layerIds.has(layer.layer_id)) {
      throw new Error("Neuron visualizations содержит пустой или повторяющийся layer_id.");
    }
    layerIds.add(layer.layer_id);
    if (!["input_filter", "max_dataset_example"].includes(layer.mode)
        || !Number.isInteger(layer.neuron_count) || layer.neuron_count < 1
        || !Array.isArray(layer.images) || layer.images.length !== layer.neuron_count) {
      throw new Error(`Некорректные изображения нейронов слоя ${layer.layer_id}.`);
    }
    for (const image of layer.images) {
      if (!Array.isArray(image) || image.length !== payload.input_shape[0]
          || image.some(row => !Array.isArray(row) || row.length !== payload.input_shape[1]
            || row.some(value => typeof value !== "number" || !Number.isFinite(value)))) {
        throw new Error(`Изображение нейрона слоя ${layer.layer_id} не совпадает с input_shape.`);
      }
    }
    if (layer.mode === "max_dataset_example"
        && (!Array.isArray(layer.source_indices)
          || layer.source_indices.length !== layer.neuron_count
          || !Array.isArray(layer.activation_values)
          || layer.activation_values.length !== layer.neuron_count)) {
      throw new Error(`Слой ${layer.layer_id} не содержит максимум для каждого нейрона.`);
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

function modelInputNode(model) {
  const firstLayer = model.layers[0];
  const shape = Array.isArray(firstLayer?.input_shape) ? firstLayer.input_shape.slice(1) : [];
  const input = node("div", "model-node model-input-node");
  input.setAttribute("aria-label", "Вход модели");
  input.append(
    node("span", "model-node-id", "input"),
    node("strong", "model-node-type", "Вход"),
    node("span", "model-node-shape", shape.length ? formatShape(shape) : "форма не сохранена"),
    node("span", "model-node-params", "данные без параметров"),
  );
  return input;
}

let currentModel = null;

function modelWeight(layerId) {
  const layer = currentModel?.layers.find(item => item.id === layerId);
  const weight = layer?.tensors?.find(item => item.name === "weight");
  return weight?.value_mode === "full" && Array.isArray(weight.values) ? weight.values : null;
}

function validateXorSnapshot(payload) {
  if (!payload || payload.schema_version !== 1 || payload.kind !== "xor_forward_snapshot") {
    throw new Error("Поддерживается только xor_forward_snapshot версии 1.");
  }
  if (payload.run_id !== runId) throw new Error("Snapshot принадлежит другому запуску.");
  if (!Array.isArray(payload.input) || payload.input.length !== 2) {
    throw new Error("XOR snapshot должен содержать два входа.");
  }
  if (!["input", "forward_hidden", "forward_output"].includes(payload.phase)) {
    throw new Error(`Неизвестная фаза snapshot: ${payload.phase || "—"}`);
  }
}

function svgNode(tag, attributes = {}) {
  const value = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, item]) => value.setAttribute(key, String(item)));
  return value;
}

function activationColor(value, available) {
  if (!available) return "#132635";
  const strength = Math.min(1, Math.abs(Number(value) || 0));
  return value < 0
    ? `rgba(255,117,129,${0.2 + strength * 0.75})`
    : `rgba(80,214,208,${0.2 + strength * 0.75})`;
}

function addXorNeuron(svg, { x, y, value, available, label, active }) {
  const group = svgNode("g", { class: active ? "xor-neuron xor-neuron-active" : "xor-neuron" });
  const circle = svgNode("circle", {
    cx: x, cy: y, r: 21,
    fill: activationColor(value, available),
  });
  const title = svgNode("title");
  title.textContent = available ? `${label}: ${formatValue(value)}` : `${label}: ещё не вычислен`;
  circle.append(title);
  const text = svgNode("text", { x, y: y + 4, "text-anchor": "middle" });
  text.textContent = available ? formatValue(value) : "·";
  const name = svgNode("text", { x, y: y + 38, "text-anchor": "middle", class: "xor-neuron-label" });
  name.textContent = label;
  group.append(circle, text, name);
  svg.append(group);
}

function addXorEdges(svg, sourceX, targetX, sourceYs, targetYs, sourceValues, weights) {
  if (!Array.isArray(weights)) return;
  const entries = [];
  targetYs.forEach((targetY, targetIndex) => {
    sourceYs.forEach((sourceY, sourceIndex) => {
      const weight = Number(weights[targetIndex]?.[sourceIndex]);
      const source = Number(sourceValues?.[sourceIndex]);
      if (!Number.isFinite(weight)) return;
      const contribution = Number.isFinite(source) ? weight * source : 0;
      entries.push({ sourceY, targetY, weight, contribution });
    });
  });
  const maximum = Math.max(...entries.map(item => Math.abs(item.contribution)), 0);
  for (const item of entries) {
    const strength = maximum > 0 ? Math.abs(item.contribution) / maximum : 0;
    const line = svgNode("line", {
      x1: sourceX + 23,
      y1: item.sourceY,
      x2: targetX - 23,
      y2: item.targetY,
      stroke: item.contribution < 0 ? "#ff7581" : "#50d6d0",
      "stroke-width": 1 + 3 * strength,
      "stroke-opacity": maximum > 0 ? 0.13 + 0.82 * strength : 0.12,
    });
    const title = svgNode("title");
    title.textContent = `weight ${formatValue(item.weight)} · activation ${formatValue(item.contribution / (item.weight || 1))} = ${formatValue(item.contribution)}`;
    line.append(title);
    svg.append(line);
  }
}

function renderXorNetwork() {
  const svg = document.querySelector("#xor-network");
  if (!svg || debugCapabilities(detail?.config)?.renderer !== "xor_neurons_v1") return;
  svg.replaceChildren();
  const hiddenWeights = modelWeight("hidden");
  const outputWeights = modelWeight("output");
  if (!hiddenWeights || !outputWeights) {
    svg.setAttribute("viewBox", "0 0 760 180");
    const message = svgNode("text", { x: 380, y: 92, "text-anchor": "middle", class: "xor-empty" });
    message.textContent = "Для схемы нужны полные малые веса model.json";
    svg.append(message);
    return;
  }

  const hiddenCount = hiddenWeights.length;
  const height = Math.max(320, hiddenCount * 62 + 80);
  const inputX = 90;
  const hiddenX = 380;
  const outputX = 670;
  const spread = (count, spacing) => Array.from({ length: count }, (_, index) =>
    height / 2 + (index - (count - 1) / 2) * spacing);
  const inputYs = spread(2, 92);
  const hiddenYs = spread(hiddenCount, Math.min(62, (height - 70) / Math.max(1, hiddenCount - 1)));
  const outputYs = [height / 2];
  svg.setAttribute("viewBox", `0 0 760 ${height}`);

  const inputValues = xorSnapshot?.input || [null, null];
  const hiddenValues = xorSnapshot?.hidden || [];
  const outputValues = xorSnapshot?.phase === "forward_output" ? xorSnapshot.post : [];
  addXorEdges(svg, inputX, hiddenX, inputYs, hiddenYs, inputValues, hiddenWeights);
  addXorEdges(svg, hiddenX, outputX, hiddenYs, outputYs, hiddenValues, outputWeights);

  inputYs.forEach((y, index) => addXorNeuron(svg, {
    x: inputX, y, value: inputValues[index], available: inputValues[index] != null,
    label: `x${index}`, active: xorSnapshot?.phase === "input",
  }));
  hiddenYs.forEach((y, index) => addXorNeuron(svg, {
    x: hiddenX, y, value: hiddenValues[index], available: hiddenValues[index] != null,
    label: `h${index}`, active: xorSnapshot?.phase === "forward_hidden",
  }));
  addXorNeuron(svg, {
    x: outputX, y: outputYs[0], value: outputValues[0], available: outputValues[0] != null,
    label: "P(XOR=1)", active: xorSnapshot?.phase === "forward_output",
  });

  for (const [x, label] of [[inputX, "Вход"], [hiddenX, "ReLU"], [outputX, "Sigmoid"]]) {
    const heading = svgNode("text", { x, y: 24, "text-anchor": "middle", class: "xor-column-label" });
    heading.textContent = label;
    svg.append(heading);
  }
}

function renderXorSnapshot(snapshot) {
  xorSnapshot = snapshot;
  const phaseNames = {
    input: "вход принят",
    forward_hidden: "вычислен скрытый слой",
    forward_output: "вычислен выход",
  };
  const result = snapshot.phase === "forward_output"
    ? ` · P(1)=${formatValue(snapshot.probability)} · класс ${snapshot.prediction}`
    : "";
  document.querySelector("#xor-forward-meta").textContent =
    `snapshot #${snapshot.seq} · ${phaseNames[snapshot.phase]} · вход [${snapshot.input.map(formatValue).join(", ")}]${result}`;
  const paused = currentControl?.requested_status === "paused";
  showNotice("#xor-debug-message", snapshot.phase === "input"
    ? paused
      ? "Вход принят. Нажмите «Один шаг», чтобы вычислить скрытый слой."
      : "Вход принят. Вычисляем скрытый слой…"
    : snapshot.phase === "forward_hidden"
      ? paused
        ? "Скрытый слой вычислен. Ещё один шаг вычислит выход."
        : "Скрытый слой вычислен. Вычисляем выход…"
      : `Forward завершён: модель предсказывает ${snapshot.prediction}. Подайте следующий вход.`);
  renderXorNetwork();
  renderXorInputControls();
}

async function loadXorEvents() {
  if (xorSnapshotLoading || debugCapabilities(detail?.config)?.renderer !== "xor_neurons_v1") return;
  xorSnapshotLoading = true;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/events?after_seq=${xorEventSeq}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    const nextEventSeq = payload.last_seq ?? xorEventSeq;
    const event = [...(payload.items || [])].reverse().find(item => item.type === "xor_forward" && item.snapshot);
    if (event) {
      const encodedPath = event.snapshot.split("/").map(encodeURIComponent).join("/");
      const snapshotResponse = await fetch(`/api/runs/${encodedRunId}/artifacts/${encodedPath}`, { cache: "no-store" });
      const snapshot = await snapshotResponse.json();
      if (!snapshotResponse.ok) throw new Error(snapshot.detail || `API ответил ${snapshotResponse.status}`);
      validateXorSnapshot(snapshot);
      renderXorSnapshot(snapshot);
    }
    xorEventSeq = nextEventSeq;
    showNotice("#xor-debug-error", "");
  } catch (error) {
    showNotice("#xor-debug-error", `Не удалось прочитать XOR snapshot: ${error.message}`);
  } finally {
    xorSnapshotLoading = false;
  }
}

function setupXorDebug(payload) {
  const panel = document.querySelector("#xor-debug-panel");
  const launch = document.querySelector("#xor-debug-launch");
  const session = document.querySelector("#xor-debug-session");
  const isXorRenderer = debugCapabilities(payload.config)?.renderer === "xor_neurons_v1";
  const artifactPaths = new Set((payload.artifacts || []).map(item => item.path));
  const canStart = payload.debug_adapter?.renderer === "xor_neurons_v1"
    && payload.manifest.status === "completed"
    && artifactPaths.has("model.json")
    && artifactPaths.has("checkpoint.pt");
  panel.classList.toggle("hidden", !isXorRenderer && !canStart);
  launch.classList.toggle("hidden", !canStart || isXorRenderer);
  session.classList.toggle("hidden", !isXorRenderer);
  if (isXorRenderer) {
    renderXorNetwork();
    loadXorEvents();
  }
}

function validateModelDebugSnapshot(snapshot) {
  if (snapshot?.schema_version !== 1 || snapshot?.kind !== "model_debug_snapshot") {
    throw new Error("неподдерживаемая версия model debug snapshot");
  }
  if (snapshot.run_id !== runId || !Number.isInteger(snapshot.seq) || snapshot.seq < 1) {
    throw new Error("model debug snapshot принадлежит другому запуску");
  }
  const input = snapshot.input;
  if (input?.mode !== "dataset_index" || !Number.isInteger(input.index)) {
    throw new Error("model debug snapshot не содержит индекс датасета");
  }
  if (!Array.isArray(input.preview) || input.preview.length !== 28
      || input.preview.some(row => !Array.isArray(row) || row.length !== 28)) {
    throw new Error("preview входа должен иметь форму 28×28");
  }
  if (!Array.isArray(snapshot.layers)) throw new Error("layers должен быть массивом");
}

function renderModelInputPreview(snapshot) {
  const canvas = document.querySelector("#model-input-preview");
  const context = canvas.getContext("2d");
  const preview = snapshot.input.preview;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / 28;
  const scaleY = canvas.height / 28;
  preview.forEach((row, y) => row.forEach((raw, x) => {
    const value = Math.max(0, Math.min(1, Number(raw) || 0));
    const shade = Math.round(value * 255);
    context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
    context.fillRect(x * scaleX, y * scaleY, Math.ceil(scaleX), Math.ceil(scaleY));
  }));
  const result = snapshot.prediction == null
    ? ""
    : ` · prediction ${snapshot.prediction}${snapshot.prediction === snapshot.input.label ? " ✓" : " ≠ label"}`;
  document.querySelector("#model-input-meta").textContent =
    `test[${snapshot.input.index}] · label ${snapshot.input.label}${result}`;
}

function tensorSummaryValues(tensor) {
  const summary = tensor?.summary || {};
  const shape = Array.isArray(tensor?.shape) ? tensor.shape.join(" × ") : "—";
  return [
    shape,
    tensor?.dtype || "—",
    formatValue(summary.min),
    formatValue(summary.max),
    formatValue(summary.mean),
    formatValue(summary.std),
    formatValue(summary.l2_norm),
    summary.sparsity == null ? "—" : `${(summary.sparsity * 100).toFixed(1)}%`,
  ];
}

function renderTensorFlowTable(layer) {
  const scroll = node("div", "tensor-flow-table-scroll");
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", `Агрегаты тензоров слоя ${layer.layer_id}`);
  scroll.tabIndex = 0;

  const table = node("table", "tensor-flow-table");
  const head = node("thead", "");
  const headRow = node("tr", "");
  ["тензор", "форма", "dtype", "min", "max", "mean", "std", "L2", "sparsity"].forEach(label => {
    headRow.append(node("th", "", label));
  });
  head.append(headRow);

  const body = node("tbody", "");
  for (const [label, tensor] of [
    ["input", layer.input], ["z", layer.preactivation], ["post", layer.output],
  ]) {
    const row = node("tr", "");
    row.append(node("th", "tensor-flow-name", label));
    tensorSummaryValues(tensor).forEach(value => row.append(node("td", "", value)));
    body.append(row);
  }
  table.append(head, body);
  scroll.append(table);
  return scroll;
}

function renderModelModuleHierarchy(layers) {
  const container = document.querySelector("#model-module-hierarchy");
  const card = document.querySelector("#model-module-card");
  container.replaceChildren();
  const nested = layers.some(layer => String(layer.module_path || "").split(".").length > 2);
  card.classList.toggle("hidden", !nested);
  if (!nested) {
    return;
  }
  const paths = new Set(["model"]);
  layers.forEach(layer => {
    let current = "model";
    String(layer.module_path || layer.layer_id).split(".").forEach(part => {
      current = `${current}.${part}`;
      paths.add(current);
    });
  });
  [...paths].forEach(path => {
    const depth = path.split(".").length - 1;
    const layer = layers.find(item => `model.${item.module_path}` === path);
    const row = node("div", layer ? "module-tree-row module-tree-leaf" : "module-tree-row");
    row.style.paddingLeft = `${depth * 18}px`;
    row.append(node("span", "mono-label", path.split(".").at(-1)));
    if (layer) row.append(node("span", "cell-sub", `${layer.layer_type} · ${layer.layer_id}`));
    container.append(row);
  });
}

function renderModelTensorFlow(snapshot) {
  const container = document.querySelector("#model-tensor-flow");
  container.replaceChildren();
  if (!snapshot.layers.length) {
    container.append(node("p", "empty", "Нет завершённых слоёв."));
    return;
  }
  snapshot.layers.forEach((layer, index) => {
    if (index) container.append(node("div", "tensor-flow-arrow", "↓"));
    const card = node("article", "tensor-flow-layer");
    const heading = node("div", "tensor-flow-heading");
    heading.append(
      node("strong", "", layer.layer_id),
      node("span", "mono-label", layer.module_path),
      node("span", "cell-sub", `${layer.layer_type} → ${layer.activation} · ${formatValue(layer.parameter_count)} параметров`),
    );
    card.append(heading, renderTensorFlowTable(layer));
    container.append(card);
  });
  if (snapshot.top_classes?.length) {
    const top = node("div", "model-top-classes");
    top.append(node("strong", "", "Top classes"));
    snapshot.top_classes.forEach(item => {
      top.append(node("span", "mono-label", `${item.class_index}: ${(item.probability * 100).toFixed(2)}%`));
    });
    container.append(top);
  }
}

function renderModelDebugSnapshot(snapshot) {
  modelDebugSnapshot = snapshot;
  renderModelInputPreview(snapshot);
  renderModelModuleHierarchy(snapshot.layers);
  renderModelTensorFlow(snapshot);
  if (currentModel) renderSelectedLayer(currentModel);
  const paused = currentControl?.requested_status === "paused";
  const complete = snapshot.prediction != null;
  showNotice("#model-debug-message", complete
    ? `Forward завершён: prediction ${snapshot.prediction}, label ${snapshot.input.label}. Выберите следующий пример.`
    : snapshot.layers.length
      ? paused
        ? `Завершено слоёв: ${snapshot.layers.length}. Нажмите «Один шаг» для следующего слоя.`
        : `Завершено слоёв: ${snapshot.layers.length}. Вычисляем следующий слой…`
      : paused
        ? "Пример принят. Нажмите «Один шаг» для первого слоя."
        : "Пример принят. Запускаем послойный forward…");
  renderModelDebugInputControls();
}

async function loadModelDebugEvents() {
  if (modelDebugSnapshotLoading || debugCapabilities(detail?.config)?.renderer !== "tensor_flow_v1") return;
  modelDebugSnapshotLoading = true;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/events?after_seq=${modelDebugEventSeq}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    const event = [...(payload.items || [])].reverse().find(item => item.type === "model_debug" && item.snapshot);
    if (event) {
      const encodedPath = event.snapshot.split("/").map(encodeURIComponent).join("/");
      const snapshotResponse = await fetch(`/api/runs/${encodedRunId}/artifacts/${encodedPath}`, { cache: "no-store" });
      const snapshot = await snapshotResponse.json();
      if (!snapshotResponse.ok) throw new Error(snapshot.detail || `API ответил ${snapshotResponse.status}`);
      validateModelDebugSnapshot(snapshot);
      renderModelDebugSnapshot(snapshot);
    }
    modelDebugEventSeq = payload.last_seq ?? modelDebugEventSeq;
    showNotice("#model-debug-error", "");
  } catch (error) {
    showNotice("#model-debug-error", `Не удалось прочитать model debug snapshot: ${error.message}`);
  } finally {
    modelDebugSnapshotLoading = false;
  }
}

function setupModelDebug(payload) {
  const panel = document.querySelector("#model-debug-panel");
  const flowPanel = document.querySelector("#model-flow-panel");
  const launch = document.querySelector("#model-debug-launch");
  const session = document.querySelector("#model-debug-session");
  const selectedInput = document.querySelector("#model-selected-input");
  const selectionWorkspace = document.querySelector("#model-selection-workspace");
  const renderer = debugCapabilities(payload.config)?.renderer;
  const isTensorFlowRenderer = renderer === "tensor_flow_v1";
  const artifactPaths = new Set((payload.artifacts || []).map(item => item.path));
  const canStart = payload.debug_adapter?.renderer === "tensor_flow_v1"
    && payload.manifest.status === "completed"
    && artifactPaths.has("model.json")
    && artifactPaths.has("checkpoint.pt");
  panel.classList.toggle("hidden", !isTensorFlowRenderer && !canStart);
  flowPanel.classList.toggle("hidden", !isTensorFlowRenderer);
  launch.classList.toggle("hidden", !canStart || isTensorFlowRenderer);
  session.classList.toggle("hidden", !isTensorFlowRenderer);
  selectedInput.classList.toggle("hidden", !isTensorFlowRenderer);
  selectionWorkspace.classList.toggle("model-selection-with-input", isTensorFlowRenderer);
  if (isTensorFlowRenderer) loadModelDebugEvents();
}

function setupRunControl(payload) {
  const isTerminal = terminalStatuses.has(payload.manifest.status);
  const supportsStep = debugCapabilities(payload.config)?.supports_step === true
    || payload.config?.experiment === "xor_backprop";
  document.querySelector("#control-panel").classList.toggle("hidden", isTerminal);
  document.querySelectorAll("[data-debug-step]").forEach(element => {
    element.classList.toggle("hidden", !supportsStep);
  });
}

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
  graph.append(modelInputNode(model));
  model.layers.forEach((layer, index) => {
    if (!index) {
      const edge = node("div", "model-edge model-edge-forward");
      edge.append(node("span", "model-edge-arrow", "→"), node("small", "", "данные"));
      graph.append(edge);
    } else {
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

function validateXorTrainingSnapshot(payload) {
  if (!payload || payload.schema_version !== 1 || payload.kind !== "xor_train_step_snapshot") {
    throw new Error("Поддерживается только xor_train_step_snapshot версии 1.");
  }
  if (payload.run_id !== runId) throw new Error("Snapshot обучения принадлежит другому запуску.");
  if (!Number.isInteger(payload.step) || payload.step < 0 || !Number.isFinite(payload.loss)) {
    throw new Error("Snapshot обучения содержит некорректные step/loss.");
  }
  if (!Array.isArray(payload.layers) || !payload.layers.length) {
    throw new Error("Snapshot обучения не содержит слои.");
  }
  for (const layer of payload.layers) {
    if (!layer.layer_id || !Array.isArray(layer.parameters) || !layer.parameters.length) {
      throw new Error("Слой snapshot обучения не содержит параметры.");
    }
    for (const parameter of layer.parameters) {
      if (!parameter.name || parameter.before == null || parameter.delta == null || parameter.after == null) {
        throw new Error("Параметр snapshot требует before/delta/after.");
      }
    }
  }
  const surface = payload.decision_surface;
  if (!surface || !Array.isArray(surface.x0) || !Array.isArray(surface.x1)
    || !Array.isArray(surface.probabilities)
    || surface.probabilities.length !== surface.x1.length) {
    throw new Error("Snapshot обучения содержит некорректную границу решений.");
  }
}

function jsonTensorShape(value) {
  const shape = [];
  let cursor = value;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    cursor = cursor[0];
  }
  return shape;
}

function renderTrainingHeatmap(selector, title, name, values) {
  const container = document.querySelector(selector);
  container.replaceChildren();
  const shape = jsonTensorShape(values);
  const section = renderTensorHeatmap(
    { name, shape },
    tensorRows(values, shape),
  );
  section.querySelector("h4").textContent = title;
  container.append(section);
}

function renderXorTrainingLoss() {
  const selected = xorTrainingEvents[xorTrainingIndex];
  const steps = xorTrainingEvents.map(event => event.step);
  const losses = xorTrainingEvents.map(event => event.scalars?.loss ?? null);
  const marker = selected ? [{ x: selected.step, y: selected.scalars?.loss }] : [];
  Plotly.react("xor-training-loss", [
    {
      x: steps, y: losses, type: "scatter", mode: "lines+markers", name: "loss",
      line: { color: "#50d6d0", width: 2 }, marker: { size: 4 },
    },
    {
      x: marker.map(item => item.x), y: marker.map(item => item.y), type: "scatter",
      mode: "markers", name: "текущий кадр", hoverinfo: "skip",
      marker: { size: 12, color: "#f3bf63", line: { color: "#fff4d0", width: 2 } },
    },
  ], {
    margin: { l: 54, r: 18, t: 15, b: 42 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#0a151f",
    font: { color: "#9bb0bd", size: 10 },
    xaxis: { title: "step", gridcolor: "#213748", zerolinecolor: "#294256" },
    yaxis: { title: "loss", type: "log", gridcolor: "#213748", zerolinecolor: "#294256" },
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

function probabilityColor(value) {
  const probability = Math.max(0, Math.min(1, Number(value) || 0));
  const low = [255, 117, 129];
  const high = [80, 214, 208];
  const mix = (left, right) => Math.round(left + (right - left) * probability);
  return `rgb(${mix(low[0], high[0])},${mix(low[1], high[1])},${mix(low[2], high[2])})`;
}

function renderDecisionBoundary(surface) {
  const canvas = document.querySelector("#xor-decision-boundary");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 20, top: 18, bottom: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const rows = surface.probabilities.length;
  const columns = surface.x0.length;
  context.clearRect(0, 0, width, height);
  const cellWidth = plotWidth / columns;
  const cellHeight = plotHeight / rows;
  surface.probabilities.forEach((row, rowIndex) => row.forEach((probability, columnIndex) => {
    context.fillStyle = probabilityColor(probability);
    context.fillRect(
      padding.left + columnIndex * cellWidth,
      padding.top + (rows - rowIndex - 1) * cellHeight,
      Math.ceil(cellWidth) + 1,
      Math.ceil(cellHeight) + 1,
    );
  }));

  context.strokeStyle = "rgba(255,255,255,.9)";
  context.lineWidth = 1.4;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const current = surface.probabilities[row][column] >= .5;
      const x = padding.left + column * cellWidth;
      const y = padding.top + (rows - row - 1) * cellHeight;
      if (column + 1 < columns && current !== (surface.probabilities[row][column + 1] >= .5)) {
        context.beginPath(); context.moveTo(x + cellWidth, y); context.lineTo(x + cellWidth, y + cellHeight); context.stroke();
      }
      if (row + 1 < rows && current !== (surface.probabilities[row + 1][column] >= .5)) {
        context.beginPath(); context.moveTo(x, y); context.lineTo(x + cellWidth, y); context.stroke();
      }
    }
  }

  const xMin = surface.x0[0];
  const xMax = surface.x0[surface.x0.length - 1];
  const yMin = surface.x1[0];
  const yMax = surface.x1[surface.x1.length - 1];
  const projectX = value => padding.left + (value - xMin) / (xMax - xMin) * plotWidth;
  const projectY = value => padding.top + plotHeight - (value - yMin) / (yMax - yMin) * plotHeight;
  [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].forEach(([x0, x1, target]) => {
    context.beginPath(); context.arc(projectX(x0), projectY(x1), 8, 0, Math.PI * 2);
    context.fillStyle = target ? "#50d6d0" : "#ff7581";
    context.fill(); context.strokeStyle = "#071019"; context.lineWidth = 3; context.stroke();
  });
  context.fillStyle = "#9bb0bd";
  context.font = "12px Inter, sans-serif";
  context.textAlign = "center";
  context.fillText("x₀", padding.left + plotWidth / 2, height - 13);
  context.save(); context.translate(15, padding.top + plotHeight / 2); context.rotate(-Math.PI / 2);
  context.fillText("x₁", 0, 0); context.restore();
}

function renderXorTrainingSnapshot(snapshot) {
  const layerSelect = document.querySelector("#xor-training-layer");
  const parameterSelect = document.querySelector("#xor-training-parameter");
  const previousLayer = layerSelect.value;
  layerSelect.replaceChildren();
  snapshot.layers.forEach(layer => {
    const option = node("option", "", layer.layer_id);
    option.value = layer.layer_id;
    layerSelect.append(option);
  });
  layerSelect.value = snapshot.layers.some(layer => layer.layer_id === previousLayer)
    ? previousLayer : snapshot.layers[0].layer_id;
  const layer = snapshot.layers.find(item => item.layer_id === layerSelect.value);
  const previousParameter = parameterSelect.value;
  parameterSelect.replaceChildren();
  layer.parameters.forEach(parameter => {
    const option = node("option", "", parameter.name);
    option.value = parameter.name;
    parameterSelect.append(option);
  });
  parameterSelect.value = layer.parameters.some(parameter => parameter.name === previousParameter)
    ? previousParameter : layer.parameters[0].name;
  const parameter = layer.parameters.find(item => item.name === parameterSelect.value);
  const prefix = parameter.name === "weight" ? "W" : "b";
  renderTrainingHeatmap("#xor-weight-before", `${prefix}_before`, parameter.name, parameter.before);
  renderTrainingHeatmap("#xor-weight-delta", `Δ${prefix}`, parameter.name, parameter.delta);
  renderTrainingHeatmap("#xor-weight-after", `${prefix}_after`, parameter.name, parameter.after);
  renderDecisionBoundary(snapshot.decision_surface);
  renderXorTrainingLoss();
  const learning = layer.apical_deviation == null
    ? " · a−baseline/e: нет у backprop"
    : " · a−baseline/e сохранены";
  showNotice("#xor-training-message",
    `Кадр #${snapshot.seq} · step ${snapshot.step} · loss ${formatValue(snapshot.loss)} · accuracy ${formatValue(snapshot.accuracy)} · ${snapshot.updated ? "параметры обновлены" : "финальная оценка без update"}${learning}`);
}

async function selectXorTrainingFrame(index) {
  if (!xorTrainingEvents.length) return;
  xorTrainingIndex = Math.max(0, Math.min(index, xorTrainingEvents.length - 1));
  const event = xorTrainingEvents[xorTrainingIndex];
  const slider = document.querySelector("#xor-training-frame");
  slider.max = String(xorTrainingEvents.length - 1);
  slider.value = String(xorTrainingIndex);
  document.querySelector("#xor-training-frame-label").textContent =
    `${xorTrainingIndex + 1} / ${xorTrainingEvents.length} · step ${event.step}`;
  document.querySelector("#xor-training-prev").disabled = xorTrainingIndex === 0;
  document.querySelector("#xor-training-next").disabled = xorTrainingIndex >= xorTrainingEvents.length - 1;
  renderXorTrainingLoss();
  try {
    let snapshot = xorTrainingSnapshots.get(event.seq);
    if (!snapshot) {
      const encodedPath = event.snapshot.split("/").map(encodeURIComponent).join("/");
      const response = await fetch(`/api/runs/${encodedRunId}/artifacts/${encodedPath}`, { cache: "no-store" });
      snapshot = await response.json();
      if (!response.ok) throw new Error(snapshot.detail || `API ответил ${response.status}`);
      validateXorTrainingSnapshot(snapshot);
      xorTrainingSnapshots.set(event.seq, snapshot);
    }
    if (xorTrainingEvents[xorTrainingIndex]?.seq === event.seq) renderXorTrainingSnapshot(snapshot);
    showNotice("#xor-training-error", "");
  } catch (error) {
    showNotice("#xor-training-error", `Не удалось прочитать train-step snapshot: ${error.message}`);
  }
}

function stopXorTrainingPlayback() {
  xorTrainingPlaying = false;
  if (xorTrainingTimer !== null) window.clearInterval(xorTrainingTimer);
  xorTrainingTimer = null;
  document.querySelector("#xor-training-play").textContent = "Воспроизвести";
}

function toggleXorTrainingPlayback() {
  if (xorTrainingPlaying) {
    stopXorTrainingPlayback();
    return;
  }
  if (xorTrainingEvents.length < 2) return;
  if (xorTrainingIndex >= xorTrainingEvents.length - 1) xorTrainingIndex = -1;
  xorTrainingPlaying = true;
  document.querySelector("#xor-training-play").textContent = "Пауза анимации";
  xorTrainingTimer = window.setInterval(() => {
    if (xorTrainingIndex >= xorTrainingEvents.length - 1) {
      stopXorTrainingPlayback();
      return;
    }
    selectXorTrainingFrame(xorTrainingIndex + 1);
  }, 450);
}

async function loadXorTrainingEvents() {
  if (xorTrainingLoading || detail?.config?.experiment !== "xor_backprop") return;
  xorTrainingLoading = true;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/events?after_seq=${xorTrainingEventSeq}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    const wasFollowing = xorTrainingIndex < 0 || xorTrainingIndex === xorTrainingEvents.length - 1;
    const known = new Set(xorTrainingEvents.map(event => event.seq));
    const additions = (payload.items || []).filter(event =>
      event.type === "xor_train_step" && event.snapshot && !known.has(event.seq));
    xorTrainingEvents.push(...additions);
    xorTrainingEvents.sort((left, right) => left.seq - right.seq);
    xorTrainingEventSeq = payload.last_seq ?? xorTrainingEventSeq;
    document.querySelector("#xor-training-frame").max = String(Math.max(0, xorTrainingEvents.length - 1));
    if (additions.length && wasFollowing && !xorTrainingPlaying) {
      await selectXorTrainingFrame(xorTrainingEvents.length - 1);
    } else if (xorTrainingEvents.length) {
      renderXorTrainingLoss();
    } else {
      const finished = terminalStatuses.has(detail?.manifest?.status);
      showNotice("#xor-training-message", finished
        ? "В этом запуске нет V.12-снимков обучающих шагов. Повторите запуск текущим кодом."
        : "Первый снимок появится после завершения атомарного train-step.");
    }
  } catch (error) {
    showNotice("#xor-training-error", `Не удалось прочитать события обучения: ${error.message}`);
  } finally {
    xorTrainingLoading = false;
  }
}

function setupXorTraining(payload) {
  const enabled = payload.config?.experiment === "xor_backprop";
  document.querySelector("#xor-training-panel").classList.toggle("hidden", !enabled);
  if (enabled) loadXorTrainingEvents();
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

function layerActivationValues(layerId) {
  const snapshotLayer = modelDebugSnapshot?.layers?.find(item => item.layer_id === layerId);
  const values = snapshotLayer?.output?.values;
  return Array.isArray(values) && values.length === 1 && Array.isArray(values[0])
    ? values[0] : null;
}

function drawNeuronImage(canvas, image, mode) {
  const height = image.length;
  const width = image[0].length;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const maximum = Math.max(0, ...image.flat().map(Math.abs));
  image.forEach((row, y) => row.forEach((value, x) => {
    if (mode === "input_filter") {
      context.fillStyle = heatColor(value, maximum);
    } else {
      const shade = Math.round(255 * Math.max(0, Math.min(1, value)));
      context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
    }
    context.fillRect(x, y, 1, 1);
  }));
}

function neuronImageStats(image) {
  const values = image.flat();
  return {
    minimum: Math.min(...values),
    maximum: Math.max(...values),
    sum: values.reduce((total, value) => total + value, 0),
  };
}

function neuronImageCard(image, mode, title, description, ariaLabel) {
  const card = node("div", "neuron-image-card");
  const canvas = node("canvas", "neuron-image-canvas");
  canvas.setAttribute("aria-label", ariaLabel);
  drawNeuronImage(canvas, image, mode);
  const caption = node("div", "neuron-image-caption");
  caption.append(node("strong", "", title), node("span", "cell-sub", description));
  card.append(canvas, caption);
  return card;
}

function renderNeuronInspector(layer, visualization) {
  const section = node("section", "neuron-layer-inspector");
  const heading = node("div", "neuron-layer-heading");
  const headingText = node("div", "");
  headingText.append(
    node("h4", "", "Нейроны слоя"),
    node("p", "cell-sub", "Порядок фиксирован; цвет показывает post для выбранного примера."),
  );
  const activationValues = layerActivationValues(layer.id);
  const finiteActivations = (activationValues || []).filter(Number.isFinite);
  const activationMaximum = Math.max(0, ...finiteActivations.map(Math.abs));
  heading.append(
    headingText,
    node("span", "mono-label", activationValues
      ? `heatmap 0 … ${formatValue(activationMaximum)}`
      : "heatmap появится после forward"),
  );

  let selectedIndex = selectedNeuronByLayer.get(layer.id) ?? 0;
  if (selectedIndex < 0 || selectedIndex >= visualization.neuron_count) selectedIndex = 0;
  selectedNeuronByLayer.set(layer.id, selectedIndex);

  const body = node("div", "neuron-layer-body");
  const matrixWrap = node("div", "neuron-matrix-wrap");
  const matrix = node("div", "neuron-matrix");
  const columns = Math.ceil(Math.sqrt(visualization.neuron_count));
  matrix.style.setProperty("--neuron-columns", String(columns));
  matrix.setAttribute("aria-label", `Нейроны слоя ${layer.id} в фиксированном порядке`);
  for (let index = 0; index < visualization.neuron_count; index += 1) {
    const activation = activationValues?.[index];
    const selected = index === selectedIndex;
    const neuronName = layer.id === "output" ? `класс ${index}` : `нейрон ${index}`;
    const button = node("button", selected ? "neuron-cell neuron-cell-selected" : "neuron-cell", String(index));
    button.type = "button";
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute("aria-label", `Выбрать ${neuronName} слоя ${layer.id}`);
    button.title = activation == null
      ? `${neuronName} · активация ещё не вычислена`
      : `${neuronName} · post ${formatValue(activation)}`;
    if (activation != null && Number.isFinite(activation)) {
      button.style.background = heatColor(activation, activationMaximum);
    }
    button.addEventListener("click", () => {
      selectedNeuronByLayer.set(layer.id, index);
      renderSelectedLayer(currentModel);
    });
    matrix.append(button);
  }
  matrixWrap.append(matrix);

  const preview = node("div", "neuron-image-panel");
  const image = visualization.images[selectedIndex];
  preview.append(node(
    "strong",
    "neuron-image-title",
    layer.id === "output" ? `Класс ${selectedIndex} · нейрон ${selectedIndex}` : `Нейрон ${selectedIndex}`,
  ));
  const imageGrid = node(
    "div",
    visualization.mode === "input_filter" ? "neuron-image-grid" : "neuron-image-grid neuron-image-grid-single",
  );
  if (visualization.mode === "input_filter") {
    const weightStats = neuronImageStats(image);
    imageGrid.append(neuronImageCard(
      image,
      "input_filter",
      "Веса",
      `min ${formatValue(weightStats.minimum)} · max ${formatValue(weightStats.maximum)}`,
      `Хитмапа входных весов нейрона ${selectedIndex} слоя ${layer.id}`,
    ));
    const input = modelDebugSnapshot?.input?.preview;
    const hasMatchingInput = Array.isArray(input) && input.length === image.length
      && input.every((row, index) => Array.isArray(row) && row.length === image[index].length);
    if (hasMatchingInput) {
      const contribution = image.map((row, y) => row.map((weight, x) => weight * input[y][x]));
      const contributionStats = neuronImageStats(contribution);
      imageGrid.append(neuronImageCard(
        contribution,
        "input_filter",
        "Вклад input × weight",
        `Σ ${formatValue(contributionStats.sum)} · без bias`,
        `Хитмапа вклада input на веса нейрона ${selectedIndex} слоя ${layer.id}`,
      ));
    } else {
      imageGrid.append(node("div", "neuron-image-unavailable cell-sub", "Вклад появится после выбора входа."));
    }
  } else {
    imageGrid.append(neuronImageCard(
      image,
      visualization.mode,
      "Максимально активирующий пример",
      `test[${visualization.source_indices[selectedIndex]}] · максимум post ${formatValue(visualization.activation_values[selectedIndex])}`,
      `Максимально активирующий test-пример нейрона ${selectedIndex} слоя ${layer.id}`,
    ));
  }
  preview.append(imageGrid);
  body.append(matrixWrap, preview);
  section.append(heading, body);
  return section;
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

  const neuronVisualization = neuronVisualizations?.layers?.find(
    item => item.layer_id === layer.id,
  );
  if (neuronVisualization) {
    inspector.append(renderNeuronInspector(layer, neuronVisualization));
  } else if (neuronVisualizationError) {
    inspector.append(node("p", "notice notice-error model-notice", neuronVisualizationError));
  }

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
  renderXorNetwork();
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

async function loadNeuronVisualizations(payload) {
  const path = "neuron-visualizations.json";
  const artifact = (payload.artifacts || []).find(item => item.path === path);
  if (!artifact) {
    if (neuronVisualizationSignature || neuronVisualizations || neuronVisualizationError) {
      neuronVisualizationSignature = "";
      neuronVisualizations = null;
      neuronVisualizationError = "";
      if (currentModel) renderSelectedLayer(currentModel);
    }
    return;
  }
  const signature = `${path}:${artifact.size_bytes}:${artifact.modified_at}`;
  if (signature === neuronVisualizationSignature) return;
  neuronVisualizationSignature = signature;
  const token = ++neuronVisualizationLoadToken;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/artifacts/${path}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    const payload = await response.json();
    validateNeuronVisualizations(payload);
    if (token !== neuronVisualizationLoadToken) return;
    neuronVisualizations = payload;
    neuronVisualizationError = "";
    if (currentModel) renderSelectedLayer(currentModel);
  } catch (error) {
    if (token !== neuronVisualizationLoadToken) return;
    neuronVisualizations = null;
    neuronVisualizationError = `Не удалось прочитать визуализации нейронов: ${error.message}`;
    if (currentModel) renderSelectedLayer(currentModel);
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
  renderDebugSessions(payload.debug_sessions || []);
  renderFinal(payload.metrics.final || {});
  renderKeyValues("#config-table", payload.config);
  renderKeyValues("#env-table", payload.metrics.env || {});
  renderKeyValues("#git-table", payload.metrics.git || {});
  renderCharts(payload.metrics);
  renderArtifacts(payload.artifacts || []);
  loadModel(payload);
  loadNeuronVisualizations(payload);
  setupRunControl(payload);
  setupXorTraining(payload);
  setupXorDebug(payload);
  setupModelDebug(payload);
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

function xorForwardPending() {
  const inputSeq = currentControl?.input_seq || 0;
  if (!inputSeq) return false;
  return !xorSnapshot
    || xorSnapshot.input_command_seq < inputSeq
    || (xorSnapshot.input_command_seq === inputSeq && xorSnapshot.phase !== "forward_output");
}

function renderXorInputControls() {
  const busy = xorInputLoading || controlCommandLoading || xorForwardPending();
  const submit = document.querySelector("#xor-set-input");
  submit.disabled = busy || !xorCanSetInput;
  submit.textContent = xorInputLoading ? "Подаём вход…" : "Подать вход";
  submit.setAttribute("aria-busy", xorInputLoading ? "true" : "false");
  document.querySelectorAll("[data-xor-input]").forEach(button => {
    button.disabled = busy || !xorCanSetInput;
  });
}

function modelDebugForwardPending() {
  const inputSeq = currentControl?.input_seq || 0;
  if (!inputSeq) return false;
  return !modelDebugSnapshot
    || modelDebugSnapshot.input_command_seq < inputSeq
    || (modelDebugSnapshot.input_command_seq === inputSeq && modelDebugSnapshot.prediction == null);
}

function renderModelDebugInputControls() {
  const busy = modelDebugInputLoading || controlCommandLoading || modelDebugForwardPending();
  const submit = document.querySelector("#model-set-example");
  submit.disabled = busy || !modelDebugCanSetInput;
  submit.textContent = modelDebugInputLoading ? "Выбираем…" : "Выбрать пример";
  submit.setAttribute("aria-busy", modelDebugInputLoading ? "true" : "false");
}

function renderControl(control) {
  currentControl = control;
  const available = new Set(control.available_commands || []);
  document.querySelectorAll("[data-run-command]").forEach(button => {
    button.disabled = controlCommandLoading || !available.has(button.dataset.runCommand);
  });
  const delay = document.querySelector("#delay-ms");
  xorCanSetInput = available.has("set_input");
  modelDebugCanSetInput = available.has("set_input");
  renderXorInputControls();
  renderModelDebugInputControls();
  if (document.activeElement !== delay) delay.value = String(control.delay_ms ?? 0);
  const actual = statusNames[control.status] || control.status;
  const requested = statusNames[control.requested_status] || control.requested_status;
  const pending = control.status === control.requested_status
    ? "" : ` · запрошено: ${requested}`;
  const lifecycle = control.lifecycle || {};
  const worker = lifecycle.worker;
  const recovery = lifecycle.recovery;
  const runtime = worker
    ? ` · ${lifecycle.pool}-worker #${worker.attempt}, heartbeat ${formatDate(worker.heartbeat_at)}`
    : ` · ${lifecycle.pool || "—"}-worker отсутствует`;
  const durable = recovery
    ? ` · recovery #${recovery.generation} (${recovery.safe_point_cursor})`
    : lifecycle.resume_unavailable_reason ? ` · resume: ${lifecycle.resume_unavailable_reason}` : "";
  showNotice("#control-message",
    `Состояние: ${actual}${pending} · задержка ${control.delay_ms ?? 0} мс · команда #${control.last_command_seq}${runtime}${durable}`);
}

async function renewActivity() {
  if (!detail || !isDebugSession(detail.config) || document.visibilityState !== "visible") return;
  if (Date.now() - lastUserInteractionAt > 20000) return;
  try {
    await fetch(`/api/runs/${encodedRunId}/activity`, { method: "POST" });
  } catch (_error) {
    // Activity heartbeat is advisory; control polling will show connectivity errors.
  }
}

async function loadControl() {
  if (controlPollLoading || controlCommandLoading) return;
  controlPollLoading = true;
  const revision = controlRevision;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/control`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    if (revision === controlRevision) {
      renderControl(payload);
      showNotice("#control-error", "");
    }
  } catch (error) {
    showNotice("#control-error", `Не удалось прочитать управление: ${error.message}`);
  } finally {
    controlPollLoading = false;
  }
}

async function issueControl(command, delayMs = null, inputValues = null) {
  if (controlCommandLoading) return false;
  controlCommandLoading = true;
  controlRevision += 1;
  if (currentControl) renderControl(currentControl);
  const body = { command };
  if (delayMs !== null) body.delay_ms = delayMs;
  if (inputValues !== null) body.input_values = inputValues;
  let succeeded = false;
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    renderControl(payload.control);
    showNotice("#control-error", "");
    fetchDetail();
    succeeded = true;
  } catch (error) {
    showNotice("#control-error", error.message);
  } finally {
    controlCommandLoading = false;
    if (currentControl) renderControl(currentControl);
  }
  if (!succeeded) loadControl();
  return succeeded;
}

async function startXorDebug() {
  const button = document.querySelector("#start-xor-debug");
  button.disabled = true;
  button.textContent = "Создаём сессию…";
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/debug`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    window.location.assign(payload.location);
  } catch (error) {
    showNotice("#xor-debug-error", error.message);
    button.disabled = false;
    button.textContent = "Открыть отладочную сессию";
  }
}

async function startModelDebug() {
  const button = document.querySelector("#start-model-debug");
  button.disabled = true;
  button.textContent = "Создаём сессию…";
  try {
    const response = await fetch(`/api/runs/${encodedRunId}/debug`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    window.location.assign(payload.location);
  } catch (error) {
    showNotice("#model-debug-error", error.message);
    button.disabled = false;
    button.textContent = "Открыть инспекцию модели";
  }
}

async function submitModelExample() {
  const value = Number(document.querySelector("#model-example-index").value);
  const debug = debugCapabilities(detail?.config) || {};
  const minimum = Number.isInteger(debug.input_min) ? debug.input_min : 0;
  const maximum = Number.isInteger(debug.input_max) ? debug.input_max : 9999;
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    showNotice("#model-debug-error", `Индекс примера должен быть целым от ${minimum} до ${maximum}.`);
    return;
  }
  if (!modelDebugCanSetInput) {
    showNotice("#model-debug-error", "Выбор примера сейчас недоступен: сессия завершена или отменяется.");
    return;
  }
  if (modelDebugForwardPending()) {
    showNotice("#model-debug-error", "Сначала завершите текущий forward: «Продолжить» или «Один шаг».");
    return;
  }
  if (modelDebugInputLoading) return;
  modelDebugInputLoading = true;
  renderModelDebugInputControls();
  try {
    if (await issueControl("set_input", null, [value])) {
      showNotice("#model-debug-error", "");
      showNotice("#model-debug-message", currentControl?.requested_status === "paused"
        ? "Пример передан. Нажмите «Один шаг» для первого слоя."
        : "Пример передан. Запускаем forward…");
      await loadModelDebugEvents();
    }
  } finally {
    modelDebugInputLoading = false;
    renderModelDebugInputControls();
  }
}

async function submitXorInput() {
  const values = ["#xor-x0", "#xor-x1"].map(selector => Number(document.querySelector(selector).value));
  if (values.some(value => !Number.isFinite(value))) {
    showNotice("#xor-debug-error", "Оба входа XOR должны быть конечными числами.");
    return;
  }
  if (!xorCanSetInput) {
    showNotice("#xor-debug-error", "Ввод сейчас недоступен: сессия завершена или отменяется.");
    return;
  }
  if (xorForwardPending()) {
    showNotice("#xor-debug-error", "Сначала завершите текущий forward: «Продолжить» или «Один шаг».");
    return;
  }
  if (xorInputLoading) return;
  xorInputLoading = true;
  renderXorInputControls();
  try {
    if (await issueControl("set_input", null, values)) {
      showNotice("#xor-debug-error", "");
      showNotice("#xor-debug-message", currentControl?.requested_status === "paused"
        ? "Вход передан. Нажмите «Один шаг» для вычисления скрытого слоя."
        : "Вход передан. Запускаем forward…");
      await loadXorEvents();
    }
  } finally {
    xorInputLoading = false;
    renderXorInputControls();
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

document.querySelector("#start-xor-debug").addEventListener("click", startXorDebug);
document.querySelector("#start-model-debug").addEventListener("click", startModelDebug);
document.querySelector("#xor-training-prev").addEventListener("click", () => {
  stopXorTrainingPlayback();
  selectXorTrainingFrame(xorTrainingIndex - 1);
});
document.querySelector("#xor-training-next").addEventListener("click", () => {
  stopXorTrainingPlayback();
  selectXorTrainingFrame(xorTrainingIndex + 1);
});
document.querySelector("#xor-training-play").addEventListener("click", toggleXorTrainingPlayback);
document.querySelector("#xor-training-frame").addEventListener("input", event => {
  stopXorTrainingPlayback();
  selectXorTrainingFrame(Number(event.target.value));
});
document.querySelector("#xor-training-layer").addEventListener("change", () => {
  const event = xorTrainingEvents[xorTrainingIndex];
  const snapshot = event && xorTrainingSnapshots.get(event.seq);
  if (snapshot) renderXorTrainingSnapshot(snapshot);
});
document.querySelector("#xor-training-parameter").addEventListener("change", () => {
  const event = xorTrainingEvents[xorTrainingIndex];
  const snapshot = event && xorTrainingSnapshots.get(event.seq);
  if (snapshot) renderXorTrainingSnapshot(snapshot);
});
document.addEventListener("pointerdown", () => { lastUserInteractionAt = Date.now(); }, { passive: true });
document.addEventListener("keydown", () => { lastUserInteractionAt = Date.now(); });
setInterval(renewActivity, 15000);
document.querySelector("#xor-input-form").addEventListener("submit", event => {
  event.preventDefault();
  submitXorInput();
});
document.querySelector("#model-example-form").addEventListener("submit", event => {
  event.preventDefault();
  submitModelExample();
});
document.querySelectorAll("[data-xor-input]").forEach(button => {
  button.addEventListener("click", () => {
    const [x0, x1] = button.dataset.xorInput.split(",");
    document.querySelector("#xor-x0").value = x0;
    document.querySelector("#xor-x1").value = x1;
    submitXorInput();
  });
});
document.querySelector("#rerun-reset").addEventListener("click", () => {
  if (rerunPreview) renderRerunForm(rerunPreview);
});
document.querySelector("#delete-run").addEventListener("click", async event => {
  if (!window.confirm(`Удалить запуск ${runId} без возможности восстановления?`)) return;
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Удаляем…";
  try {
    const response = await fetch(`/api/runs/${encodedRunId}`, { method: "DELETE" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    window.location.assign("/runs");
  } catch (error) {
    showNotice("#page-error", `Не удалось удалить запуск: ${error.message}`);
    button.disabled = false;
    button.textContent = "Удалить";
  }
});
document.querySelectorAll("[data-run-command]:not([data-run-command='set_delay'])").forEach(button => {
  button.addEventListener("click", () => {
    const command = button.dataset.runCommand;
    if (command === "cancel"
      && !window.confirm("Отменить этот запуск в следующей безопасной точке?")) {
      return;
    }
    issueControl(command);
  });
});
document.querySelector("#delay-form").addEventListener("submit", event => {
  event.preventDefault();
  const delayMs = Number(document.querySelector("#delay-ms").value);
  if (!Number.isInteger(delayMs) || delayMs < 0 || delayMs > 60000) {
    showNotice("#control-error", "Задержка должна быть целым числом от 0 до 60000 мс.");
    return;
  }
  issueControl("set_delay", delayMs);
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
setInterval(() => {
  if (!detail || !terminalStatuses.has(detail.manifest.status)) loadControl();
}, 1000);
setInterval(loadXorEvents, 500);
setInterval(loadModelDebugEvents, 500);
setInterval(loadXorTrainingEvents, 500);
fetchDetail();
loadControl();
pollLog(true);
loadRerunPreview();
