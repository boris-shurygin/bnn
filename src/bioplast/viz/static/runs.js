"use strict";

const form = document.querySelector("#run-filters");
const body = document.querySelector("#runs-body");
const caption = document.querySelector("#catalog-caption");
const errorBox = document.querySelector("#catalog-error");
const warningPanel = document.querySelector("#catalog-warnings");
const warningList = document.querySelector("#catalog-warning-list");
const refreshButton = document.querySelector("#refresh-runs");
const autoRefresh = document.querySelector("#auto-refresh");
const showDebugRuns = document.querySelector("#show-debug-runs");
const experimentFilter = document.querySelector("#experiment-filter");
const pageSize = document.querySelector("#page-size");
const previousPage = document.querySelector("#previous-page");
const nextPage = document.querySelector("#next-page");
const pageIndicator = document.querySelector("#page-indicator");
const selectPageRuns = document.querySelector("#select-page-runs");
const deleteSelectedRuns = document.querySelector("#delete-selected-runs");
const selectedRunIds = new Set();
const knownExperiments = new Set();
let offset = 0;
let loading = false;
let reloadRequested = false;
let deleting = false;
let currentPayload = null;

const statusNames = {
  queued: "в очереди", running: "в работе", paused: "пауза",
  completed: "завершён", failed: "ошибка", cancelled: "отменён",
};
const terminalStatuses = new Set(["completed", "failed", "cancelled"]);

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
  params.set("include_debug", String(showDebugRuns.checked));
  params.set("offset", String(offset));
  params.set("limit", pageSize.value);
  return params;
}

function renderStats(payload) {
  const counts = payload.counts || {};
  document.querySelector("#stat-total").textContent = payload.total;
  document.querySelector("#stat-completed").textContent = counts.completed || 0;
  document.querySelector("#stat-running").textContent =
    (counts.running || 0) + (counts.queued || 0) + (counts.paused || 0);
  document.querySelector("#stat-failed").textContent = counts.failed || 0;
}

function updateSelectionControls() {
  const selectable = (currentPayload?.items || []).filter(run => terminalStatuses.has(run.status));
  const selectedOnPage = selectable.filter(run => selectedRunIds.has(run.run_id)).length;
  selectPageRuns.disabled = selectable.length === 0 || deleting;
  selectPageRuns.checked = selectable.length > 0 && selectedOnPage === selectable.length;
  selectPageRuns.indeterminate = selectedOnPage > 0 && selectedOnPage < selectable.length;
  deleteSelectedRuns.disabled = selectedRunIds.size === 0 || deleting;
  deleteSelectedRuns.textContent = selectedRunIds.size
    ? `Удалить выбранные (${selectedRunIds.size})` : "Удалить выбранные";
}

function renderPagination(payload) {
  const pages = Math.max(1, Math.ceil(payload.total / payload.limit));
  const page = Math.min(pages, Math.floor(payload.offset / payload.limit) + 1);
  pageIndicator.textContent = `Страница ${page} из ${pages}`;
  previousPage.disabled = payload.offset === 0 || loading;
  nextPage.disabled = payload.offset + payload.items.length >= payload.total || loading;
}

function makeSelectionCell(run, row) {
  const cell = node("td", "selection-cell");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.disabled = !terminalStatuses.has(run.status);
  checkbox.checked = selectedRunIds.has(run.run_id);
  checkbox.setAttribute("aria-label", `Выбрать ${run.run_id}`);
  if (checkbox.disabled) checkbox.title = "Активный запуск сначала нужно завершить";
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) selectedRunIds.add(run.run_id);
    else selectedRunIds.delete(run.run_id);
    row.classList.toggle("run-selected", checkbox.checked);
    updateSelectionControls();
  });
  cell.append(checkbox);
  return cell;
}

function renderRuns(payload) {
  currentPayload = payload;
  body.replaceChildren();
  renderStats(payload);
  const first = payload.items.length ? payload.offset + 1 : 0;
  const last = payload.offset + payload.items.length;
  caption.textContent = `Показано ${first}–${last} из ${payload.total}`;
  if (!payload.items.length) {
    const row = node("tr");
    const cell = node("td", "empty", "По этим фильтрам запусков нет.");
    cell.colSpan = 7;
    row.append(cell);
    body.append(row);
  }
  for (const experiment of payload.experiments || []) knownExperiments.add(experiment);
  for (const run of payload.items) {
    if (run.experiment) knownExperiments.add(run.experiment);
    const row = node("tr", selectedRunIds.has(run.run_id) ? "run-selected" : "");
    const identity = node("td");
    const link = node("a", "run-link", run.run_id);
    link.href = `/runs/${encodeURIComponent(run.run_id)}`;
    identity.append(link);
    if (run.is_debug) identity.append(node("span", "debug-badge", "debug"));
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

    const actions = node("td", "run-actions");
    const deleteButton = node("button", "button button-danger run-delete-button", "Удалить");
    deleteButton.type = "button";
    deleteButton.disabled = !terminalStatuses.has(run.status);
    deleteButton.title = deleteButton.disabled ? "Активный запуск сначала нужно завершить" : "Удалить запуск";
    deleteButton.addEventListener("click", () => deleteRuns([run.run_id]));
    actions.append(deleteButton);

    row.append(makeSelectionCell(run, row), identity, state, model, result, started, actions);
    body.append(row);
  }
  const selectedExperiment = experimentFilter.value;
  const allExperiments = document.createElement("option");
  allExperiments.value = "";
  allExperiments.textContent = "Все";
  experimentFilter.replaceChildren(allExperiments, ...[...knownExperiments].sort().map(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
  experimentFilter.value = selectedExperiment;

  warningList.replaceChildren();
  warningPanel.classList.toggle("hidden", !payload.errors.length);
  for (const item of payload.errors) {
    warningList.append(node("li", "", `${item.run_id}: ${item.error}`));
  }
  renderPagination(payload);
  updateSelectionControls();
}

async function loadRuns() {
  if (loading) {
    reloadRequested = true;
    return;
  }
  loading = true;
  refreshButton.disabled = true;
  previousPage.disabled = true;
  nextPage.disabled = true;
  errorBox.classList.add("hidden");
  try {
    const response = await fetch(`/api/runs?${queryFromForm()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`API ответил ${response.status}`);
    const payload = await response.json();
    if (!payload.items.length && payload.total > 0 && offset > 0) {
      offset = Math.max(0, Math.floor((payload.total - 1) / payload.limit) * payload.limit);
      const retry = await fetch(`/api/runs?${queryFromForm()}`, { cache: "no-store" });
      if (!retry.ok) throw new Error(`API ответил ${retry.status}`);
      renderRuns(await retry.json());
    } else {
      renderRuns(payload);
    }
  } catch (error) {
    errorBox.textContent = `Не удалось прочитать каталог: ${error.message}`;
    errorBox.classList.remove("hidden");
  } finally {
    loading = false;
    refreshButton.disabled = false;
    if (currentPayload) renderPagination(currentPayload);
    if (reloadRequested) {
      reloadRequested = false;
      loadRuns();
    }
  }
}

function resetCatalogView() {
  offset = 0;
  selectedRunIds.clear();
  updateSelectionControls();
}

async function deleteRuns(runIds) {
  if (deleting || !runIds.length) return;
  const label = runIds.length === 1 ? `запуск ${runIds[0]}` : `${runIds.length} запусков`;
  if (!window.confirm(`Удалить ${label} без возможности восстановления?`)) return;
  deleting = true;
  errorBox.classList.add("hidden");
  updateSelectionControls();
  try {
    const response = await fetch("/api/runs", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_ids: runIds }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `API ответил ${response.status}`);
    for (const runId of payload.deleted) selectedRunIds.delete(runId);
    await loadRuns();
  } catch (error) {
    errorBox.textContent = `Не удалось удалить: ${error.message}`;
    errorBox.classList.remove("hidden");
  } finally {
    deleting = false;
    updateSelectionControls();
  }
}

form.addEventListener("submit", event => { event.preventDefault(); resetCatalogView(); loadRuns(); });
form.addEventListener("reset", () => { resetCatalogView(); setTimeout(loadRuns, 0); });
showDebugRuns.addEventListener("change", () => { resetCatalogView(); loadRuns(); });
pageSize.addEventListener("change", () => { resetCatalogView(); loadRuns(); });
previousPage.addEventListener("click", () => { offset = Math.max(0, offset - Number(pageSize.value)); loadRuns(); });
nextPage.addEventListener("click", () => { offset += Number(pageSize.value); loadRuns(); });
selectPageRuns.addEventListener("change", () => {
  for (const run of currentPayload?.items || []) {
    if (!terminalStatuses.has(run.status)) continue;
    if (selectPageRuns.checked) selectedRunIds.add(run.run_id);
    else selectedRunIds.delete(run.run_id);
  }
  renderRuns(currentPayload);
});
deleteSelectedRuns.addEventListener("click", () => deleteRuns([...selectedRunIds]));
refreshButton.addEventListener("click", loadRuns);
setInterval(() => { if (autoRefresh.checked) loadRuns(); }, 5000);
loadRuns();
