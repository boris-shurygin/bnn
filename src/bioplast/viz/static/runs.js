"use strict";

const form = document.querySelector("#run-filters");
const body = document.querySelector("#runs-body");
const caption = document.querySelector("#catalog-caption");
const errorBox = document.querySelector("#catalog-error");
const warningPanel = document.querySelector("#catalog-warnings");
const warningList = document.querySelector("#catalog-warning-list");
const refreshButton = document.querySelector("#refresh-runs");
const autoRefresh = document.querySelector("#auto-refresh");
let loading = false;

const statusNames = {
  queued: "в очереди", running: "в работе", paused: "пауза",
  completed: "завершён", failed: "ошибка", cancelled: "отменён",
};

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium", timeStyle: "medium",
  }).format(date);
}

function formatValue(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .001)
      ? value.toExponential(2) : value.toLocaleString("ru-RU", { maximumFractionDigits: 4 });
  }
  return String(value);
}

function statusBadge(status) {
  return node("span", `status status-${status}`, statusNames[status] || status);
}

function queryFromForm() {
  const params = new URLSearchParams();
  for (const [key, raw] of new FormData(form).entries()) {
    if (!raw) continue;
    let value = raw;
    if ((key === "started_after" || key === "started_before") && raw) {
      value = new Date(raw).toISOString();
    }
    params.set(key, value);
  }
  params.set("limit", "500");
  return params;
}

function renderStats(items) {
  const count = status => items.filter(item => item.status === status).length;
  document.querySelector("#stat-total").textContent = items.length;
  document.querySelector("#stat-completed").textContent = count("completed");
  document.querySelector("#stat-running").textContent = count("running") + count("queued") + count("paused");
  document.querySelector("#stat-failed").textContent = count("failed");
}

function renderRuns(payload) {
  body.replaceChildren();
  renderStats(payload.items);
  caption.textContent = `Показано ${payload.items.length} из ${payload.total}`;
  if (!payload.items.length) {
    const row = node("tr");
    const cell = node("td", "empty", "По этим фильтрам запусков нет.");
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
  const experiments = new Set();
  for (const run of payload.items) {
    if (run.experiment) experiments.add(run.experiment);
    const row = node("tr");
    const identity = node("td");
    const link = node("a", "run-link", run.run_id);
    link.href = `/runs/${encodeURIComponent(run.run_id)}`;
    identity.append(link);
    identity.append(node("div", "cell-sub", [run.dataset, `seed ${run.seed ?? "—"}`].filter(Boolean).join(" · ")));

    const state = node("td");
    state.append(statusBadge(run.status));
    if (run.dirty) state.append(node("span", "dirty-badge", "dirty"));

    const model = node("td");
    model.append(node("div", "", run.model || "—"));
    model.append(node("div", "cell-sub", run.experiment || "—"));

    const result = node("td");
    const pills = node("div", "metric-pills");
    for (const [key, value] of Object.entries(run.final || {}).slice(0, 4)) {
      if (["number", "boolean", "string"].includes(typeof value) && value !== null) {
        pills.append(node("span", "metric-pill", `${key}: ${formatValue(value)}`));
      }
    }
    result.append(pills.childElementCount ? pills : node("span", "cell-sub", "нет итоговых метрик"));

    const started = node("td");
    started.append(node("div", "", formatDate(run.started_at)));
    const duration = run.duration_sec == null ? "ещё идёт" : `${formatValue(run.duration_sec)} с`;
    started.append(node("div", "cell-sub", duration));
    row.append(identity, state, model, result, started);
    body.append(row);
  }
  const datalist = document.querySelector("#experiment-options");
  datalist.replaceChildren(...[...experiments].sort().map(value => {
    const option = document.createElement("option");
    option.value = value;
    return option;
  }));

  warningList.replaceChildren();
  warningPanel.classList.toggle("hidden", !payload.errors.length);
  for (const item of payload.errors) {
    warningList.append(node("li", "", `${item.run_id}: ${item.error}`));
  }
}

async function loadRuns() {
  if (loading) return;
  loading = true;
  refreshButton.disabled = true;
  errorBox.classList.add("hidden");
  try {
    const response = await fetch(`/api/runs?${queryFromForm()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    renderRuns(await response.json());
  } catch (error) {
    errorBox.textContent = `Не удалось прочитать каталог: ${error.message}`;
    errorBox.classList.remove("hidden");
  } finally {
    loading = false;
    refreshButton.disabled = false;
  }
}

form.addEventListener("submit", event => { event.preventDefault(); loadRuns(); });
form.addEventListener("reset", () => setTimeout(loadRuns, 0));
refreshButton.addEventListener("click", loadRuns);
setInterval(() => { if (autoRefresh.checked) loadRuns(); }, 5000);
loadRuns();
