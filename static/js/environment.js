/* Environment: live tiles (/api/environment) + forecast charts (/api/forecast). */
const TILE_ICONS = { temperature: "thermometer", feels_like: "thermometer-sun", humidity: "droplets", rainfall: "cloud-rain",
                     wind_speed: "wind", uv_index: "sun", pm25: "circle-gauge", pm10: "haze", aqi: "leaf", heat_wave_days: "flame" };
const TEMP_KEYS = new Set(["temperature", "feels_like"]);

function tileValue(t, v) {
  if (v === null || v === undefined) return "–";
  if (TEMP_KEYS.has(t.key)) return fmt.temp(v, 0);
  const d = t.key === "rainfall" ? 1 : 0;
  return fmt.num(v, d);
}
function tileUnit(t, v = t.value) {
  if (TEMP_KEYS.has(t.key)) return fmt.tempUnit();
  if (t.key === "heat_wave_days") return Number(v) === 1 ? "day" : "days";
  return t.unit;
}

function renderTiles(e) {
  document.getElementById("env-observed").textContent =
    `${e.location} · ${e.live ? "observed" : "last available"} ${e.observed_at ? fmt.time(e.observed_at) : ""}`;
  document.getElementById("env-tiles").innerHTML = e.tiles.map((t) => {
    const lvl = t.level;
    const color = lvl ? RISK[lvl].c : cssVar("--muted");
    const arrow = t.trend ? `<i data-lucide="${{ up: "arrow-up", down: "arrow-down", flat: "minus" }[t.trend]}" class="w-3.5 h-3.5 text-[var(--accent)]" data-tip="vs 24 hours earlier"></i>` : "";
    const tag = t.used_by_model
      ? `<span class="chip !cursor-default">Used by model</span>`
      : `<span class="chip !cursor-default" style="background:var(--border-soft);color:var(--body)">Display only</span>`;
    let modelLine = "";
    if (t.used_by_model && t.model_value !== null && t.model_value !== undefined) {
      const mv = TEMP_KEYS.has(t.key) ? `${fmt.temp(t.model_value, 1)}${fmt.tempUnit()}`
        : t.key === "rainfall" ? `${fmt.num(t.model_value, 1)} mm (7-day total)`
        : `${fmt.num(t.model_value, t.key === "heat_wave_days" ? 0 : 1)} ${tileUnit(t, t.model_value)}`;
      modelLine = `<div class="text-[10.5px] mt-1 text-[var(--muted)]">Model uses 7-day ${t.key === "rainfall" || t.key === "heat_wave_days" ? "" : "mean "}value: ${mv}</div>`;
    }
    return `
      <div class="panel p-3 relative overflow-hidden" style="border-left:4px solid ${color}">
        <div class="flex items-center justify-between gap-1">
          <span class="flex items-center gap-1.5 text-[12px] text-[var(--heading)] font-medium"><i data-lucide="${TILE_ICONS[t.key]}" class="w-4 h-4" style="color:${color}"></i>${esc(t.label)}</span>
          ${arrow}
        </div>
        <div class="mt-2 flex items-baseline gap-1"><span class="text-[24px] font-bold text-[var(--heading)] leading-none">${tileValue(t, t.value)}</span><span class="text-[12px]">${esc(tileUnit(t))}</span></div>
        <div class="text-[11px] mt-1.5 leading-snug" style="color:${lvl === "Moderate" ? "#B7791F" : color}">${esc(t.meaning)}</div>
        ${modelLine}
        <div class="mt-2">${tag}</div>
      </div>`;
  }).join("");
}

function wrapError(id, msg, retry) {
  const wrap = document.getElementById(id);
  wrap.innerHTML = `<div class="empty-state h-full flex flex-col items-center justify-center gap-2">
    <i data-lucide="cloud-off" class="w-6 h-6"></i><div>Forecast unavailable: ${esc(msg)}</div>
    <button class="btn-ghost">Retry</button></div>`;
  wrap.querySelector("button").onclick = retry;
  lucide.createIcons();
}

function renderForecast(f) {
  const labels = f.days.map((d) => new Date(d.date + "T00:00:00").toLocaleDateString("en-US", { weekday: "short", day: "numeric" }));
  const conv = (v) => (SETTINGS.units === "F" ? v * 9 / 5 + 32 : v);
  new Chart(document.getElementById("fc-chart"), {
    data: { labels, datasets: [
      { type: "line", label: `Max ${fmt.tempUnit()}`, data: f.days.map((d) => conv(d.tmax)), borderColor: RISK.High.c,
        backgroundColor: RISK.High.c + "22", fill: "+1", tension: 0.3, pointRadius: 3, yAxisID: "t" },
      { type: "line", label: `Min ${fmt.tempUnit()}`, data: f.days.map((d) => conv(d.tmin)), borderColor: "#1F6FEB",
        backgroundColor: "#1F6FEB", tension: 0.3, pointRadius: 3, yAxisID: "t" },
      { type: "bar", label: "Rain (mm)", data: f.days.map((d) => d.rain), backgroundColor: "rgba(21,151,187,.35)",
        borderRadius: 3, yAxisID: "r" },
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, color: cssVar("--body") } },
                 tooltip: { callbacks: { afterBody: (i) => (f.days[i[0].dataIndex].heatwave ? "Heat-wave day (above threshold)" : "") } } },
      scales: {
        t: { position: "left", title: { display: true, text: fmt.tempUnit(), color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
        r: { position: "right", beginAtZero: true, title: { display: true, text: "mm", color: cssVar("--muted") }, grid: { display: false } },
        x: { grid: { display: false } },
      },
    },
  });

  const hl = f.hours.map((h) => new Date(h.time).toLocaleString("en-US", { weekday: "short", hour: "numeric" }));
  const thr = f.thresholds;
  const line = (value, label, color, axis) => ({ type: "line", label, data: f.hours.map(() => value), borderColor: color,
                                                  borderDash: [6, 4], borderWidth: 1.2, pointRadius: 0, yAxisID: axis });
  new Chart(document.getElementById("aq-chart"), {
    data: { labels: hl, datasets: [
      { type: "line", label: "PM2.5 (µg/m³)", data: f.hours.map((h) => h.pm25), borderColor: "#8E5BD0", backgroundColor: "#8E5BD0",
        borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: "pm" },
      { type: "line", label: "US AQI", data: f.hours.map((h) => h.aqi), borderColor: "#0E4C7A", backgroundColor: "#0E4C7A",
        borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: "aqi" },
      line(thr.who_pm25_24h, `WHO PM2.5 24-h guideline (${thr.who_pm25_24h} µg/m³)`, "#8E5BD0", "pm"),
      line(thr.aqi_usg, `AQI ${thr.aqi_usg} — sensitive groups`, RISK.Moderate.c, "aqi"),
      line(thr.aqi_unhealthy, `AQI ${thr.aqi_unhealthy} — unhealthy`, RISK.High.c, "aqi"),
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 }, color: cssVar("--body") } } },
      scales: {
        pm: { position: "left", beginAtZero: true, title: { display: true, text: "PM2.5 µg/m³", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
        aqi: { position: "right", beginAtZero: true, suggestedMax: 160, title: { display: true, text: "US AQI", color: cssVar("--muted") }, grid: { display: false } },
        x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { display: false } },
      },
    },
  });

  const n = f.heatwave_days;
  const lvl = n >= 3 ? "High" : n >= 1 ? "Moderate" : "Low";
  document.getElementById("hw-box").innerHTML = `
    <div class="text-center shrink-0">
      <div class="text-[40px] font-bold leading-none" style="color:${RISK[lvl].c}">${n}</div>
      <div class="text-[11.5px] mt-1">of the next 7 days</div>
    </div>
    <div class="flex-1 min-w-0">
      <div class="text-[13px] text-[var(--heading)]">Days forecast above the heat-wave threshold of <b>${fmt.temp(f.heatwave_tmax, 1)}${fmt.tempUnit()}</b> (local 95th percentile).</div>
      <div class="grid grid-cols-7 gap-2 mt-2">${f.days.map((d, i) => `
        <div class="panel py-1.5 text-center ${d.heatwave ? "" : ""}" style="${d.heatwave ? `border-color:${RISK.High.c};background:var(--adv-high)` : ""}">
          <div class="text-[10.5px]">${labels[i]}</div>
          <div class="text-[13px] font-semibold text-[var(--heading)]">${fmt.temp(d.tmax, 0)}°</div>
          <div class="text-[10px]" style="color:${d.heatwave ? RISK.High.c : "var(--muted)"}">${d.heatwave ? "heat wave" : "—"}</div>
        </div>`).join("")}</div>
    </div>`;
}

async function loadForecast() {
  ["fc-wrap", "aq-wrap"].forEach((id) => {
    const w = document.getElementById(id);
    if (!w.querySelector("canvas")) w.innerHTML = `<canvas id="${id === "fc-wrap" ? "fc-chart" : "aq-chart"}"></canvas>`;
  });
  try {
    renderForecast(await apiLoc("/api/forecast"));
  } catch (e) {
    wrapError("fc-wrap", e.message, loadForecast);
    wrapError("aq-wrap", e.message, loadForecast);
    document.getElementById("hw-box").innerHTML = `<div class="empty-state w-full">Heat-wave watch needs the forecast.</div>`;
  }
  lucide.createIcons();
}

Page.run(() => apiLoc("/api/environment"), (e) => { fillHeader(e); renderTiles(e); });
loadForecast();
