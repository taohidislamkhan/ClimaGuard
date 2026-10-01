/* How It Works: every number comes from /api/methodology (artifacts/*.json). */
const MODEL_NAMES = { logreg: "Logistic Regression", rf: "Random Forest", xgb: "XGBoost", dt: "Decision Tree", elasticnet: "ElasticNet" };
const num = (v) => Number(v).toLocaleString("en-US");
let M = null;
let seasonScope = "bgd";

function renderDataset(d) {
  const set = (id, v) => { document.getElementById(id).textContent = v; };
  set("ds-rows", num(d.rows));
  set("ds-countries", d.countries);
  set("ds-regions", d.regions);
  set("ds-cols", d.columns);
  set("ds-range", `${d.start} → ${d.end}`);
  set("ds-weeks", d.weeks_per_country);
  set("ds-nulls", d.nulls);
  set("ds-dups", d.duplicate_rows);
  document.getElementById("ds-range").style.fontSize = "15px";
  document.getElementById("ds-regions-list").textContent = `Regions: ${d.region_names.join(", ")}.`;
}

function renderCleaning(c) {
  const dis = Object.values(c.negative_disease_values).reduce((a, b) => a + b, 0);
  const disAfter = Object.values(c.negative_disease_values_after).reduce((a, b) => a + b, 0);
  const rows = [
    ["Negative AQI values clamped to 0", c.negative_aqi, c.negative_aqi_after],
    ["Healthcare-access scores clipped to 100", c.healthcare_over_100, c.healthcare_over_100_after],
    ["Disease rates and counts clipped to ≥ 0", dis, disAfter],
    ["Duplicate country-week rows removed (ISO week collision)", c.duplicate_year_week_rows, 0],
  ];
  document.getElementById("cleaning-rows").innerHTML = rows.map(([r, a, b]) =>
    `<tr><td>${r}</td><td><b>${num(a)}</b>${a === 0 ? ' <span class="text-[var(--muted)]">(none in this dataset)</span>' : ""}</td><td>${num(b)}</td></tr>`).join("");
}

function corrColor(v) {
  const t = Math.min(1, Math.abs(v));
  const base = v >= 0 ? [229, 72, 77] : [31, 111, 235];
  return `rgba(${base.join(",")},${(0.08 + 0.85 * t).toFixed(2)})`;
}

function renderCorr(c) {
  const n = c.labels.length;
  let html = `<div class="grid gap-[2px]" style="grid-template-columns: 110px repeat(${n}, minmax(26px, 1fr));">`;
  html += `<div></div>` + c.labels.map((l) => `<div class="text-[9.5px] text-center leading-tight h-[84px] flex items-end justify-center" title="${esc(l)}"><span style="writing-mode:vertical-rl;transform:rotate(180deg)">${esc(l)}</span></div>`).join("");
  c.matrix.forEach((row, i) => {
    html += `<div class="text-[10.5px] text-right pr-2 self-center truncate text-[var(--heading)]">${esc(c.labels[i])}</div>`;
    html += row.map((v, j) => `<div class="heat-cell" style="background:${corrColor(v)};color:${Math.abs(v) > 0.55 ? "#fff" : "var(--heading)"}"
      data-tip="${esc(c.labels[i])} × ${esc(c.labels[j])}: r = ${v.toFixed(2)}">${i === j ? "" : (Math.abs(v) < 0.05 ? "0" : v.toFixed(1))}</div>`).join("");
  });
  document.getElementById("corr").innerHTML = html + "</div>";
}

function renderScatter(e) {
  new Chart(document.getElementById("scatter"), {
    data: {
      datasets: [
        { type: "scatter", label: "Weekly observations", data: e.scatter.temp.map((t, i) => ({ x: t, y: e.scatter.heat[i] })),
          pointRadius: 1.6, backgroundColor: "rgba(31,111,235,.25)" },
        { type: "line", label: "Mean per 2° bin", data: e.bins.mid.map((m, i) => ({ x: m, y: e.bins.mean[i] })),
          borderColor: RISK.High.c, backgroundColor: RISK.High.c, borderWidth: 2.5, pointRadius: 2, tension: 0.25 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 10, color: cssVar("--body") } } },
      scales: {
        x: { type: "linear", title: { display: true, text: "Dataset temperature", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
        y: { title: { display: true, text: "Heat-related admissions", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
      },
    },
  });
}

const SERIES_COLORS = { respiratory: "#E5484D", vector: "#22A06B", heat: "#F5A524", waterborne: "#1597BB", cardio: "#8E5BD0" };

function renderSeasonal() {
  const e = M.eda;
  const s = e.seasonal_index[seasonScope];
  const c = Chart.getChart("seasonal");
  if (c) c.destroy();
  new Chart(document.getElementById("seasonal"), {
    type: "line",
    data: { labels: e.months, datasets: Object.entries(s).map(([k, v]) => ({
      label: DISEASE_LABELS[k], data: v, borderColor: SERIES_COLORS[k], backgroundColor: SERIES_COLORS[k],
      borderWidth: 2, pointRadius: 2, tension: 0.3 })) },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "right", labels: { boxWidth: 10, color: cssVar("--body") } } },
      scales: { y: { title: { display: true, text: "Index (annual mean = 100)", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
                x: { grid: { display: false } } },
    },
  });
}

function renderFeatures(f) {
  const fam = Object.entries(f.families).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...fam.map(([, n]) => n));
  document.getElementById("families").innerHTML = fam.map(([name, n]) => `
    <div class="text-right font-semibold text-[var(--heading)]">${n}</div>
    <div><div class="h-[22px] rounded-md flex items-center px-2 text-white text-[11.5px] font-medium" style="width:${Math.max(28, 100 * n / max)}%;background:var(--primary)">${esc(name)}</div></div>`).join("");
  const top = f.funnel[0].n;
  document.getElementById("funnel").innerHTML = f.funnel.map((s, i) => `
    <div class="flex items-center gap-3">
      <div class="h-[28px] rounded-md grid place-items-center text-white text-[12px] font-semibold shrink-0 mx-auto"
           style="width:${Math.max(18, 100 * s.n / top)}%;background:${["#0E4C7A", "#155E91", "#1B77A8", "#1597BB", "#22A06B"][i] || "#1597BB"}">${s.n}</div>
    </div>
    <div class="text-[11px] text-center -mt-1">${esc(s.stage)}</div>`).join("");
  document.getElementById("methods").innerHTML = f.methods.map((m) =>
    `<li><b class="text-[var(--heading)]">${esc(m.name)}</b> — ${esc(m.detail)}</li>`).join("");
}

function renderSplit(s) {
  const start = new Date(M.dataset.start), end = new Date(s.test_end);
  const trainEnd = new Date(s.train_end), valEnd = new Date(s.val_end);
  const span = end - start;
  const seg = [
    ["Train", start, trainEnd, "#0E4C7A", s.sizes.train],
    ["Validation", trainEnd, valEnd, "#F5A524", s.sizes.val],
    ["Test", valEnd, end, "#22A06B", s.sizes.test],
  ];
  document.getElementById("timeline").innerHTML = seg.map(([name, a, b, c, n]) =>
    `<div class="flex flex-col items-center justify-center" style="width:${100 * (b - a) / span}%;background:${c}">
       <span>${name}</span><span class="text-[10.5px] font-normal opacity-90">${num(n)} rows</span></div>`).join("");
  document.getElementById("timeline-axis").innerHTML =
    [M.dataset.start, `≤ ${s.train_end}`, `≤ ${s.val_end}`, s.test_end].map((t) => `<span>${t}</span>`).join("");
}

function renderClassifiers(m) {
  const rows = m.rows.filter((r) => r.task === "overall_classifier");
  const models = [...new Set(rows.map((r) => r.model))];
  const pick = (model, split) => rows.find((r) => r.model === model && r.split === split);
  const winner = m.winners.overall_classifier;
  document.getElementById("clf-rows").innerHTML = models.map((mo) => {
    const v = pick(mo, "val"), t = pick(mo, "test");
    return `<tr class="${mo === winner ? "hi" : ""}"><td>${MODEL_NAMES[mo] || mo}${mo === winner ? " ★ <span class='chip ml-1'>selected</span>" : ""}</td>
      <td>${v.macro_f1.toFixed(3)}</td><td>${t.accuracy.toFixed(3)}</td><td>${t.macro_f1.toFixed(3)}</td><td>${t.roc_auc_ovr_weighted.toFixed(3)}</td></tr>`;
  }).join("");
  const metrics = [["accuracy", "Accuracy", "#0E4C7A"], ["macro_f1", "Macro-F1", "#1597BB"], ["roc_auc_ovr_weighted", "ROC-AUC", "#F5A524"]];
  new Chart(document.getElementById("clf-chart"), {
    type: "bar",
    data: { labels: models.map((mo) => ({ logreg: "Log. Reg.", rf: "Rand. Forest", xgb: "XGBoost", dt: "Dec. Tree" }[mo] || mo) + (mo === winner ? " ★" : "")),
            datasets: metrics.map(([k, l, c]) => ({ label: l, data: models.map((mo) => pick(mo, "test")[k]), backgroundColor: c, borderRadius: 3 })) },
    options: { maintainAspectRatio: false,
               plugins: { legend: { labels: { boxWidth: 10, color: cssVar("--body") } },
                          title: { display: true, text: "Test split (2024–2025)", color: cssVar("--muted") } },
               scales: { y: { min: 0.5, max: 1, grid: { color: cssVar("--grid") } },
                         x: { grid: { display: false }, ticks: { maxRotation: 0, font: { size: 11 } } } } },
  });
}

function renderRegressors(regs) {
  new Chart(document.getElementById("r2-chart"), {
    type: "bar",
    data: { labels: regs.map((r) => `${r.label} (${r.model})`),
            datasets: [{ data: regs.map((r) => r.r2), backgroundColor: regs.map((r) => RISK[r.reliability.level].c), borderRadius: 4 }] },
    options: { indexAxis: "y", maintainAspectRatio: false,
               plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `Test R² = ${c.raw.toFixed(3)}` } } },
               scales: { x: { min: Math.min(0, ...regs.map((r) => r.r2)) - 0.05, max: 1, title: { display: true, text: "Test R²", color: cssVar("--muted") }, grid: { color: cssVar("--grid") } },
                         y: { grid: { display: false }, ticks: { color: cssVar("--heading") } } } },
  });
  document.getElementById("r2-notes").innerHTML = regs.map((r) => `
    <div class="panel p-2.5 ${r.key === "cardio" ? "!border-[color:var(--high)]" : ""}">
      <div class="flex items-center gap-2 flex-wrap">
        <b class="text-[var(--heading)] text-[12.5px]">${esc(r.label)}</b>
        <span class="text-[11.5px]">R² ${r.r2.toFixed(3)} · RMSE ${r.rmse.toFixed(2)}</span>
        ${badge(r.reliability.level, r.reliability.label)}
      </div>
      <p class="text-[11.5px] mt-1 leading-snug">${esc(r.interpretation)}</p>
    </div>`).join("");
  const cardio = regs.find((r) => r.key === "cardio");
  if (cardio) document.getElementById("lim-cardio").textContent = cardio.r2.toFixed(3);
}

function renderShap(shap) {
  const order = ["overall", "respiratory", "vector", "heat", "waterborne", "cardio"];
  document.getElementById("shap-n").textContent = `${shap.overall.n_rows} sampled`;
  const AR = new Set(["Recent waterborne cases", "Recent heat admissions"]);
  document.getElementById("shap-grid").innerHTML = order.map((k) => `
    <div class="panel p-3">
      <div class="flex items-center justify-between gap-2"><b class="text-[var(--heading)] text-[12.5px]">${DISEASE_LABELS[k]}</b>
        <span class="text-[10.5px]">${esc(shap[k].model)}</span></div>
      <div class="mt-2" id="shap-${k}"></div>
      <div class="text-[10px] mt-2 text-[var(--muted)]">Units: mean |SHAP| in ${esc(shap[k].unit)}${k === "cardio" ? ". ElasticNet shrank most coefficients to exactly zero." : ""}</div>
    </div>`).join("");
  order.forEach((k) => {
    const g = shap[k].grouped.slice(0, 6);
    const top = g[0].mean_abs || 1;
    Components.barList(document.getElementById(`shap-${k}`), g.map((x) => ({
      label: x.label + (AR.has(x.label) ? " ⟲" : ""), value: Math.round(100 * x.mean_abs / top),
      valueText: x.mean_abs < 0.1 ? x.mean_abs.toFixed(3) : x.mean_abs.toFixed(2),
      tip: AR.has(x.label) ? "Past values of this indicator (autoregressive feature)" : null,
    })));
  });
}

function renderHealth(h) {
  const body = document.getElementById("health-body");
  if (!h) {
    body.innerHTML = `<p class="text-[12px]">No drift report yet. Run <code>dvc repro drift</code>.</p>`;
    return;
  }
  document.getElementById("health-status").innerHTML = h.retrain_recommended
    ? badge("High", "Retraining recommended") : badge("Low", "Healthy — no retraining needed");
  const d = h.data, c = h.concept, t = h.thresholds;
  const worst = d.worst.map((w) => `<tr class="${w.psi >= t.psi_threshold ? "hi" : ""}"><td>${esc(w.feature)}</td>
    <td>${w.psi.toFixed(3)}</td><td>${w.ks_pvalue < 0.001 ? "&lt; 0.001" : w.ks_pvalue.toFixed(3)}</td></tr>`).join("");
  const flagged = Object.entries(c.flagged_by_task || {});
  body.innerHTML = `
    <div class="grid gap-4" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);">
      <div class="panel p-3">
        <div class="text-[13px] font-semibold text-[var(--heading)]">Data drift (inputs)</div>
        <p class="text-[12px] mt-1"><b>${d.n_drifted}</b> of ${d.n_features} model features drifted
          (PSI ≥ ${t.psi_threshold} and KS p &lt; ${t.ks_pvalue}); <b>${d.top_features_drifted}</b> of the top
          ${d.top_features_checked} SHAP features.</p>
        <table class="tbl mt-2"><thead><tr><th>Largest shifts</th><th>PSI</th><th>KS p</th></tr></thead><tbody>${worst}</tbody></table>
      </div>
      <div class="panel p-3">
        <div class="text-[13px] font-semibold text-[var(--heading)]">Concept drift (performance)</div>
        <p class="text-[12px] mt-1">Each test quarter vs the same quarter of the validation year:
          flagged if macro-F1 drops &gt; ${t.f1_tolerance} or RMSE rises &gt; ${Math.round(100 * t.rmse_tolerance)}%.</p>
        <p class="text-[12px] mt-2"><b>${c.n_flagged}</b> of ${c.n_windows} task-quarters flagged${flagged.length
          ? ": " + flagged.map(([k, n]) => `${esc(k)} (${n})`).join(", ") : "."}</p>
      </div>
    </div>
    <div class="panel p-3 mt-3 text-[12px]">
      <b class="text-[var(--heading)]">Response policy:</b> ${esc(h.response)}
      ${h.reasons.length ? `<ul class="list-disc pl-5 mt-1">${h.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : ""}
    </div>`;
}

function renderAdvisory(rows) {
  document.getElementById("adv-rows").innerHTML = rows.map((r) => `
    <tr><td class="font-semibold">${esc(r.disease)}</td><td>${esc(r.trigger)}</td><td>${esc(r.title)}</td>
    <td><ul class="list-disc pl-4">${r.actions.map((a) => `<li class="text-[12px]">${esc(a)}</li>`).join("")}</ul></td></tr>`).join("");
}

function renderPersonalEngine(pe) {
  const lbl = { respiratory: "Respiratory", heat: "Heat-related", cardio: "Cardiovascular", vector: "Vector-borne", waterborne: "Waterborne" };
  document.getElementById("pe-matrix").innerHTML = ["Low", "Moderate", "High"].map((band) => `
    <tr><td class="font-semibold">${band}</td>${["normal", "elevated", "high"].map((s) => `<td>${badge(pe.matrix[band][s])}</td>`).join("")}</tr>`).join("");
  const o = pe.overrides;
  document.getElementById("pe-overrides").textContent =
    `for a person with HIGH respiratory or cardio sensitivity, PM2.5 > ${o.pm25_ugm3} µg/m³ or AQI > ${o.aqi} → at least Moderate, ` +
    `and PM2.5 > ${o.pm25_unhealthy} or AQI > ${o.aqi_unhealthy} → at least High (US EPA AirNow). For HIGH heat sensitivity, ` +
    `today's forecast max ≥ ${pe.heat_threshold.toFixed(0)} °C (the pipeline's heat-wave threshold) → at least Moderate.`;
  document.getElementById("pe-sources").innerHTML = pe.sources.map((x) =>
    `<li><a class="text-[var(--primary)] underline" href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.label)}</a></li>`).join("");
  document.getElementById("pe-count").textContent = pe.rules.length;
  document.getElementById("pe-rules").innerHTML = pe.rules.map((r) => `
    <tr><td class="font-semibold">${esc(lbl[r.disease])}</td><td>${esc(r.condition)}</td>
      <td><span class="badge ${r.sensitivity === "high" ? "lvl-High" : "lvl-Moderate"}">${esc(r.sensitivity)}</span></td>
      <td class="text-[12px]"><a href="${esc(r.url)}" target="_blank" rel="noopener" data-tip="${esc(r.url)}">${esc(r.source)}</a></td></tr>`).join("");
}

function setupNav() {
  const links = [...document.querySelectorAll("#section-nav a")];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.dataset.sec === e.target.id));
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  document.querySelectorAll(".doc-section").forEach((s) => obs.observe(s));
  links[0].classList.add("active");
}

Components.tabs("season-scope", (k) => { seasonScope = k; if (M) renderSeasonal(); });
setupNav();
initHeaderOnly();
Page.run(() => api("/api/methodology"), (m) => {
  M = m;
  renderDataset(m.dataset);
  renderCleaning(m.cleaning);
  renderCorr(m.correlation);
  renderScatter(m.eda);
  renderSeasonal();
  renderFeatures(m.features);
  renderSplit(m.split);
  renderClassifiers(m.metrics);
  renderRegressors(m.regressors);
  renderShap(m.shap);
  renderHealth(m.drift);
  renderAdvisory(m.advisory_table);
  renderPersonalEngine(m.personal_engine);
});
