/* Risk History: dataset-period trend, model check on the test split, seasonality, snapshots. */
const COLORS = { overall: "#0E4C7A", respiratory: "#E5484D", vector: "#22A06B", heat: "#F5A524",
                 waterborne: "#1597BB", cardio: "#8E5BD0" };
const hidden = new Set();          // series hidden via the legend survive range changes
let range = "12w", target = "respiratory", scope = "bgd";

function levelOf(v) { return v >= 65 ? "High" : v >= 40 ? "Moderate" : "Low"; }

function renderTrend(h) {
  const ds = h.dataset;
  document.getElementById("rh-span").textContent =
    `${ds.dates[0]} → ${ds.dates.at(-1)} · ${ds.dates.length} weeks · ${ds.source}`;
  const c = Chart.getChart("rh-chart");
  if (c) c.destroy();
  const keys = ["overall", "respiratory", "vector", "heat", "waterborne", "cardio"];
  const dense = ds.dates.length > 60;
  new Chart(document.getElementById("rh-chart"), {
    type: "line",
    data: { labels: ds.dates, datasets: keys.map((k) => ({
      label: DISEASE_LABELS[k], data: ds[k], borderColor: COLORS[k], backgroundColor: COLORS[k],
      borderWidth: k === "overall" ? 2.8 : 1.6, pointRadius: dense ? 0 : 2.5, cubicInterpolationMode: "monotone",
      hidden: hidden.has(k), _key: k })) },
    options: {
      maintainAspectRatio: false, animation: { duration: 300 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, color: cssVar("--body") },
                  onClick: (e, item, legend) => {
                    const k = legend.chart.data.datasets[item.datasetIndex]._key;
                    hidden.has(k) ? hidden.delete(k) : hidden.add(k);
                    Chart.defaults.plugins.legend.onClick(e, item, legend);
                  } },
        tooltip: { callbacks: { afterTitle: (i) => `split: ${ds.split[i[0].dataIndex]}` } },
      },
      scales: {
        y: { min: 0, max: 100, ticks: { stepSize: 20 }, grid: { color: cssVar("--grid") },
             title: { display: true, text: "Score (0–100)", color: cssVar("--muted") } },
        x: { grid: { display: false }, ticks: { maxTicksLimit: 12, maxRotation: 0 } },
      },
    },
    plugins: [Components.riskBands],
  });
}

function renderSnapshots(h) {
  document.getElementById("snap-chip").innerHTML = h.demo
    ? chip("Demo Data", "'backfill' rows are model runs on the last dataset weeks (2025), added so trends work before real daily snapshots accumulate.") : "";
  document.getElementById("snap-csv").href = `/api/history.csv?loc=${encodeURIComponent(h.location)}`;
  const rows = h.snapshots.slice().reverse();
  document.getElementById("snap-rows").innerHTML = rows.length ? rows.map((r) => `
    <tr><td>${r.date}</td><td>${r.source === "live" ? '<span class="chip">live</span>' : '<span class="text-[var(--muted)]">backfill</span>'}</td>
    <td><b>${Math.round(r.overall)}</b></td><td>${badge(r.overall_level)}</td>
    ${["respiratory", "vector", "heat", "waterborne", "cardio"].map((k) => `<td><span class="txt-lvl lvl-${levelOf(r[k])}">${Math.round(r[k])}</span></td>`).join("")}
    <td>${fmt.num(r.pm25, 1)}</td><td>${fmt.temp(r.temperature, 1)}${fmt.tempUnit()}</td></tr>`).join("")
    : `<tr><td colspan="11" class="empty-state">No snapshots yet.</td></tr>`;
}

async function loadHistory() {
  return Page.run(() => apiLoc(`/api/history?range=${range}`), (h) => {
    renderTrend(h);
    renderSnapshots(h);
  });
}

/* ---------------- model check ---------------- */
async function loadModelCheck() {
  let d;
  try {
    d = await api(`/api/test-predictions?target=${target}`);
  } catch (e) {
    document.getElementById("mc-stats").innerHTML = `<span class="up">Couldn't load: ${esc(e.message)}</span>`;
    return;
  }
  const rel = d.r2 >= 0.8 ? ["Low", "High reliability"] : d.r2 >= 0.4 ? ["Moderate", "Moderate reliability"]
    : ["High", "Low reliability — interpret with caution"];
  document.getElementById("mc-stats").innerHTML =
    `<b class="text-[var(--heading)]">${esc(d.model)}</b><span>R² ${d.r2.toFixed(3)}</span><span>RMSE ${d.rmse.toFixed(2)}</span>
     <span>n = ${d.n.toLocaleString()} test weeks (${d.start} → ${d.end})</span>${badge(rel[0], rel[1])}`;
  const all = d.scatter.actual.concat(d.scatter.pred);
  const lo = Math.min(...all), hi = Math.max(...all);
  const col = COLORS[target];
  [["mc-scatter"], ["mc-series"]].forEach(([id]) => { const c = Chart.getChart(id); if (c) c.destroy(); });
  new Chart(document.getElementById("mc-scatter"), {
    data: { datasets: [
      { type: "scatter", label: "Test weeks", data: d.scatter.actual.map((a, i) => ({ x: a, y: d.scatter.pred[i] })),
        pointRadius: 1.8, backgroundColor: col + "55" },
      { type: "line", label: "Perfect prediction", data: [{ x: lo, y: lo }, { x: hi, y: hi }],
        borderColor: cssVar("--muted"), borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 },
    ] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
               scales: { x: { type: "linear", title: { display: true, text: "Actual", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
                         y: { title: { display: true, text: "Predicted", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } } } },
  });
  new Chart(document.getElementById("mc-series"), {
    type: "line",
    data: { labels: d.bgd.dates, datasets: [
      { label: "Actual", data: d.bgd.actual, borderColor: cssVar("--muted"), backgroundColor: cssVar("--muted"), borderWidth: 1.4, pointRadius: 0 },
      { label: "Predicted", data: d.bgd.pred, borderColor: col, backgroundColor: col, borderWidth: 2, pointRadius: 0 },
    ] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
               plugins: { legend: { position: "bottom", labels: { boxWidth: 10, color: cssVar("--body") } } },
               scales: { x: { ticks: { maxTicksLimit: 5, maxRotation: 0 }, grid: { display: false } }, y: { grid: { color: cssVar("--grid") } } } },
  });
}

/* ---------------- seasonality heatmap ---------------- */
async function loadSeasonality() {
  const box = document.getElementById("ss-grid");
  let s;
  try {
    s = await api(`/api/seasonality?scope=${scope}`);
  } catch (e) {
    box.innerHTML = `<div class="empty-state">Couldn't load: ${esc(e.message)}</div>`;
    return;
  }
  let html = `<div class="grid gap-[3px]" style="grid-template-columns: 104px repeat(12, minmax(30px, 1fr));">`;
  html += `<div></div>` + s.months.map((m) => `<div class="text-[10.5px] text-center">${m}</div>`).join("");
  s.rows.forEach((r) => {
    html += `<div class="text-[11.5px] text-[var(--heading)] self-center">${esc(r.label)}</div>`;
    html += r.values.map((v, i) => {
      const lvl = levelOf(v);
      const alpha = Math.round(40 + 180 * Math.min(1, Math.abs(v - 50) / 50)).toString(16).padStart(2, "0");
      return `<div class="heat-cell" style="background:${RISK[lvl].c}${alpha};color:${Math.abs(v - 50) > 30 ? "#fff" : "var(--heading)"}"
        data-tip="${esc(r.label)}, ${s.months[i]}: average percentile ${v.toFixed(0)} (${lvl})">${v.toFixed(0)}</div>`;
    }).join("");
  });
  box.innerHTML = html + "</div>";
  document.getElementById("ss-note").textContent =
    `Metric: ${s.metric}. Colour uses the same bands as the scores (<40 Low, 40–64 Moderate, ≥65 High).` +
    (scope === "all" ? " Mixing both hemispheres flattens seasonal peaks." : "");
}

/* ---------------- wiring ---------------- */
document.getElementById("rh-range").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  document.querySelectorAll("#rh-range button").forEach((x) => x.classList.toggle("active", x === b));
  range = b.dataset.range;
  loadHistory();
});
Components.tabs("mc-target", (k) => { target = k; loadModelCheck(); });
Components.tabs("ss-scope", (k) => { scope = k; loadSeasonality(); });

initHeaderOnly();
loadHistory();
loadModelCheck();
loadSeasonality();
