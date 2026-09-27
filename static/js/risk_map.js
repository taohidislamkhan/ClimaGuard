/* Risk Map: choropleth of divisions (live) or dataset countries (test weeks). */
const state = { view: "division", layer: "overall", week: null, weeks: [], items: {}, selected: null };
const geo = {};
let map, baseLayer, dataLayer, labelLayer;

async function geojson(name) {
  if (!geo[name]) geo[name] = await (await fetch(`/static/geo/${name}.geojson`)).json();
  return geo[name];
}

function colorFor(level) {
  return level ? RISK[level].c : cssVar("--map-fill");
}

function initMap() {
  map = L.map("rm-map", { zoomSnap: 0.25, attributionControl: false, worldCopyJump: true });
  L.control.attribution({ prefix: false, position: "bottomleft" })
    .addAttribution("Boundaries © geoBoundaries (CC BY 4.0) · Natural Earth").addTo(map);
  map.zoomControl.setPosition("topright");
}

function clearLayers() {
  [baseLayer, dataLayer, labelLayer].forEach((l) => l && map.removeLayer(l));
  baseLayer = dataLayer = labelLayer = null;
}

async function drawDivision(data) {
  clearLayers();
  const byId = Object.fromEntries(data.items.map((i) => [i.id, i]));
  dataLayer = L.geoJSON(await geojson("bd_divisions"), {
    style: (f) => ({ color: cssVar("--card"), weight: 1.5, fillColor: colorFor(byId[f.properties.name]?.level), fillOpacity: 0.72 }),
    onEachFeature: (f, layer) => {
      const it = byId[f.properties.name];
      layer.bindTooltip(`<b>${esc(f.properties.name)}</b><br>${it ? `${esc(it.level)} · ${it.value}/100` : "no data"}`, { sticky: true });
      layer.on("click", () => openDrawer("division", f.properties.name));
    },
  }).addTo(map);
  labelLayer = L.layerGroup(data.items.map((i) => L.marker([i.lat, i.lon], {
    interactive: false,
    icon: L.divIcon({ className: "", html: `<div class="map-label-chip">${esc(i.name)} <b>${i.value}</b></div>`, iconSize: null }),
  }))).addTo(map);
  fitView();
}

async function drawCountry(data) {
  clearLayers();
  const byId = Object.fromEntries(data.items.map((i) => [i.id, i]));
  baseLayer = L.geoJSON(await geojson("world_110"), {
    filter: (f) => !byId[f.properties.code],
    style: () => ({ color: cssVar("--card"), weight: 0.6, fillColor: cssVar("--map-fill"), fillOpacity: 0.55 }),
    interactive: false,
  }).addTo(map);
  dataLayer = L.geoJSON(await geojson("world_110"), {
    filter: (f) => !!byId[f.properties.code],
    style: (f) => ({ color: cssVar("--card"), weight: 0.8, fillColor: colorFor(byId[f.properties.code].level), fillOpacity: 0.8 }),
    onEachFeature: (f, layer) => {
      const it = byId[f.properties.code];
      layer.bindTooltip(`<b>${esc(it.name)}</b><br>${it.level ? `${esc(it.level)} · ${it.value}/100` : "no data this week"}`, { sticky: true });
      layer.on("click", () => { openDrawer("country", f.properties.code); keepVisible(layer.getBounds().getCenter()); });
    },
  }).addTo(map);
  if (!state.fitted) {
    fitView();
    state.fitted = true;
  }
}

/** Fit the data so neither the control panel (left) nor an open drawer (right) hides it. */
function fitView() {
  if (!dataLayer) return;
  const drawerOpen = document.getElementById("rm-drawer").classList.contains("open");
  const bounds = state.view === "division" ? dataLayer.getBounds() : L.latLngBounds([[-45, -125], [62, 150]]);
  map.fitBounds(bounds, { paddingTopLeft: [330, 24], paddingBottomRight: [drawerOpen ? 360 : 24, 24] });
}

/** Pan so a clicked country is not hidden under the open drawer (340px) or the panel. */
function keepVisible(latlng) {
  const pt = map.latLngToContainerPoint(latlng);
  const w = map.getSize().x;
  const right = w - 380, left = 340;
  if (pt.x > right) map.panBy([pt.x - right + 20, 0]);
  else if (pt.x < left) map.panBy([pt.x - left - 20, 0]);
}

function updateChrome(data) {
  const division = state.view === "division";
  document.getElementById("rm-week-box").classList.toggle("hidden", division);
  document.getElementById("rm-chip").classList.toggle("hidden", !division);
  document.getElementById("rm-legend-na").classList.toggle("hidden", division);
  document.getElementById("rm-source-text").textContent = division
    ? `Live Open-Meteo weather per division · updated ${fmt.time(data.updated_at)}`
    : "Model predictions on the dataset's test weeks (unseen during training)";
  document.getElementById("rm-legend-title").textContent =
    `${DISEASE_LABELS[state.layer]} risk${division ? "" : ` · week of ${data.week}`}`;
  if (!division) {
    const slider = document.getElementById("rm-week");
    state.weeks = data.weeks;
    slider.max = data.weeks.length - 1;
    slider.value = data.week_index;
    document.getElementById("rm-week-label").textContent = data.week;
    document.getElementById("rm-week-ends").innerHTML = `<span>${data.weeks[0]}</span><span>${data.weeks.at(-1)}</span>`;
  }
}

function load() {
  const q = new URLSearchParams({ view: state.view, layer: state.layer });
  if (state.view === "country" && state.week !== null) q.set("week", state.week);
  return Page.run(() => apiLoc(`/api/map?${q}`), async (data) => {
    fillHeader(data);
    state.itemsById = Object.fromEntries(data.items.map((i) => [i.id, i]));
    updateChrome(data);
    if (state.view === "division") await drawDivision(data);
    else await drawCountry(data);
    if (state.selected && state.selected.view === state.view) openDrawer(state.view, state.selected.id);
  });
}

/* ---------------- drawer ---------------- */
function closeDrawer() {
  const d = document.getElementById("rm-drawer");
  d.classList.remove("open");
  d.setAttribute("aria-hidden", "true");
  state.selected = null;
  fitView();
}

async function openDrawer(view, id) {
  state.selected = { view, id };
  const drawer = document.getElementById("rm-drawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.getElementById("dr-kind").textContent = view === "division" ? "Division (live, demo)" : "Country (dataset)";
  document.getElementById("dr-name").textContent = view === "division" ? id : (state.itemsById?.[id]?.name || id);
  document.getElementById("dr-source").textContent = "";
  if (state.view === "division") fitView();
  const body = document.getElementById("dr-body");
  body.innerHTML = `<div class="space-y-2">${'<div class="skeleton-block" style="height:14px"></div>'.repeat(6)}</div>`;
  const q = new URLSearchParams({ view, id });
  if (view === "country" && state.week !== null) q.set("week", state.week);
  try {
    const d = await api(`/api/map/detail?${q}`);
    document.getElementById("dr-name").textContent = d.name;
    document.getElementById("dr-source").textContent = d.week ? `${d.source} · week of ${d.week}` : d.source;
    body.innerHTML = `
      <div class="panel p-3 flex items-center justify-between">
        <div><div class="text-[11px]">Overall score</div><div class="text-[24px] font-bold text-[var(--heading)]">${d.overall.score}</div></div>
        ${badge(d.overall.level, `${d.overall.level} risk`)}
      </div>
      <div class="text-[12px] font-semibold text-[var(--heading)] mt-4">Disease scores</div>
      <div class="mt-2 space-y-1.5">${d.diseases.map((x) => `
        <div class="flex items-center justify-between text-[12px]"><span>${esc(x.label)}</span>
          <span class="flex items-center gap-2"><b class="text-[var(--heading)]">${x.score}</b>${badge(x.level)}</span></div>`).join("")}</div>
      <div class="text-[12px] font-semibold text-[var(--heading)] mt-4">Top 3 drivers (overall model, local SHAP)</div>
      <ol class="mt-2 space-y-1 text-[12px]">${d.drivers.map((x, i) => `
        <li class="flex items-center gap-2"><span class="text-[var(--muted)]">${i + 1}.</span>
          <span class="flex-1" ${x.autoregressive ? 'data-tip="Past values of this indicator (autoregressive feature)"' : ""}>${esc(x.label)}${x.autoregressive ? " ⟲" : ""}</span>
          <span class="${x.direction === "raises" ? "up" : "down"} text-[11px]">${x.direction === "raises" ? "▲ raises" : "▼ lowers"}</span></li>`).join("")}</ol>
      <div class="text-[12px] font-semibold text-[var(--heading)] mt-4">Advisories</div>
      <div class="mt-2 space-y-1.5" id="dr-adv"></div>
      <div class="disclaimer flex gap-2 items-start text-[11px] mt-3"><i data-lucide="shield-alert" class="w-4 h-4 shrink-0"></i><span>${esc(d.disclaimer)}</span></div>`;
    Components.advisories(document.getElementById("dr-adv"), d.advisories);
    lucide.createIcons();
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Couldn't load details: ${esc(e.message)}</div>`;
  }
}

/* ---------------- controls ---------------- */
document.getElementById("rm-view").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b || b.dataset.view === state.view) return;
  document.querySelectorAll("#rm-view button").forEach((x) => x.classList.toggle("active", x === b));
  state.view = b.dataset.view;
  state.fitted = false;
  closeDrawer();
  load();
});
Components.tabs("rm-layer", (key) => { state.layer = key; load(); });
let weekTimer = null;
document.getElementById("rm-week").addEventListener("input", (ev) => {
  state.week = Number(ev.target.value);
  document.getElementById("rm-week-label").textContent = state.weeks[state.week] || "–";
  clearTimeout(weekTimer);
  weekTimer = setTimeout(load, 180);
});
document.getElementById("dr-close").addEventListener("click", closeDrawer);

initMap();
load();
