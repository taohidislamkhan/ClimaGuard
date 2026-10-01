/* My Risk: per-disease deep dive from /api/disease/<key>. Tab kept in the URL hash. */
const KEYS = ["respiratory", "vector", "heat", "waterborne", "cardio"];
let current = KEYS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "respiratory";

function setActiveTab() {
  document.querySelectorAll("#disease-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.key === current));
}

function renderTop(d) {
  const o = d.overall;
  Charts.gauge("mr-gauge", o.score, o.level);
  document.getElementById("mr-gauge-score").textContent = o.score;
  const b = document.getElementById("mr-badge");
  b.className = `badge lvl-${o.level}`;
  b.textContent = `${o.level} risk`;
  const top = [...d.tabs].sort((a, b) => b.score - a.score)[0];
  document.getElementById("mr-summary").textContent =
    `Overall environmental risk in ${d.location} is ${o.level.toLowerCase()} (${o.score}/100). ` +
    `The highest disease score today is ${top.label.toLowerCase()} (${top.score}/100). ` +
    `Scores are percentiles of the 2015–2022 training weeks: <40 low, 40–64 moderate, ≥65 high.`;
  document.getElementById("mr-updated").textContent =
    `${d.live ? "Live" : "Last available"} data · updated ${fmt.time(d.updated_at)}`;
  d.tabs.forEach((t) => {
    const el = document.querySelector(`[data-tab-score="${t.key}"]`);
    const lvl = t.score >= 65 ? "High" : t.score >= 40 ? "Moderate" : "Low";
    el.className = `badge !h-[18px] !px-[6px] !text-[10px] lvl-${lvl}`;
    el.textContent = t.score;
  });
}

function renderBreakdown(d) {
  const a = d.adjustment;
  document.getElementById("sb-model").textContent = d.model_score;
  document.getElementById("sb-model-band").innerHTML = badge(d.model_level);
  document.getElementById("sb-adj").textContent = `${a.points > 0 ? "+" : ""}${a.points}`;
  document.getElementById("sb-cap").textContent = a.capped ? `capped from +${a.raw_points} (max ±10)` : "max ±10";
  document.getElementById("sb-final").textContent = d.score;
  document.getElementById("sb-final-band").innerHTML = badge(d.level);
  document.getElementById("sb-rules").innerHTML = a.reasons.length
    ? `<div class="font-medium text-[var(--heading)]">Rules that fired:</div><ul class="list-disc pl-5 mt-1">${a.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`
    : `<span class="text-[var(--muted)]">No profile rule affects this disease. <a class="text-[var(--primary)] font-medium" href="${Loc.link("/my-health")}">Review your profile →</a></span>`;
  document.getElementById("sb-pred").textContent =
    `${d.model} predicts ${d.prediction} (${d.target}); that is the ${d.model_score}th percentile of training weeks. The profile never enters the model.`;
}

function renderReliability(d) {
  const r = d.reliability;
  document.getElementById("rel-r2").textContent = r.r2 === null ? "–" : `R² ${r.r2.toFixed(3)}`;
  const b = document.getElementById("rel-badge");
  b.className = `badge lvl-${r.level}`;
  b.textContent = r.label;
  document.getElementById("rel-model").textContent = d.model;
  document.getElementById("rel-text").textContent = d.interpretation;
}

function renderSpark(d) {
  const s = d.sparkline;
  const demo = s.sources.some((x) => x === "backfill");
  document.getElementById("spark-chip").innerHTML = demo
    ? chip("Demo Data", "Earlier points are backfilled model runs on the last dataset weeks (2025), not daily history.") : "";
  const lvl = d.level;
  const c = Chart.getChart("mr-spark");
  if (c) c.destroy();
  new Chart(document.getElementById("mr-spark"), {
    type: "line",
    data: { labels: s.labels, datasets: [{ data: s.values, borderColor: RISK[lvl].c, backgroundColor: RISK[lvl].c + "22",
                                           fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: RISK[lvl].c }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false },
               tooltip: { callbacks: { title: (i) => `${s.labels[i[0].dataIndex]} · ${s.sources[i[0].dataIndex]}` } } },
               scales: { x: { display: false }, y: { min: 0, max: 100, ticks: { stepSize: 50, font: { size: 9 } }, grid: { color: cssVar("--grid") } } } },
    plugins: [Components.riskBands],
  });
  const first = s.labels[0], last = s.labels.at(-1);
  document.getElementById("spark-axis").innerHTML = `<span>${esc(first || "")}</span><span>${esc(last || "")}</span>`;
}

function renderShap(d) {
  document.getElementById("shap-sub").textContent =
    `Top ${d.shap.length} grouped features of the ${d.model} model for ${d.location}, in ${d.target} units.` +
    (d.shap.length < 8 ? ` Only ${d.shap.length} feature group${d.shap.length === 1 ? " has" : "s have"} a non-zero contribution.` : "");
  Components.signedBars("mr-shap", d.shap, d.shap_unit);
}

function renderAdvisory(d) {
  const a = d.advisory;
  const box = document.getElementById("adv-box");
  if (!a.rule) {
    box.innerHTML = `<div class="panel p-3 text-[12px]"><b class="text-[var(--heading)]">${esc(a.title)}.</b> ${esc(a.note)}</div>`;
    return;
  }
  const status = a.fires
    ? `<span class="badge lvl-High">Active</span> <span class="text-[12px]">score ${d.model_score} ≥ ${a.threshold}</span>`
    : `<span class="badge lvl-Low">Not triggered</span> <span class="text-[12px]">score ${d.model_score} &lt; ${a.threshold}</span>`;
  box.innerHTML = `
    <div class="flex items-center gap-2">${status}</div>
    <div class="adv adv-lg tone-${a.fires ? "high" : "low"} mt-2">
      <div class="adv-title">${esc(a.title)}</div>
      <ul>${a.bullets.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
      ${a.fires ? "" : '<div class="text-[11px] mt-1 text-[var(--muted)]">Shown for reference — these actions apply when the rule fires.</div>'}
    </div>`;
}

function renderPersonal(d) {
  const p = d.personal;
  const box = document.getElementById("pa-box");
  if (!p.personalized) {
    box.innerHTML = `<div class="panel p-3 text-[12px]">No profile saved yet. <a class="text-[var(--primary)] underline" href="${Loc.link("/my-health")}">Fill in My Health</a>
      to see your personal level, the rules that apply to you and what to do.</div>`;
    return;
  }
  const sensTip = { normal: "No rule raised your sensitivity for this disease.",
                    elevated: "At least one rule marks you as somewhat more sensitive.",
                    high: "At least one rule marks you as a sensitive group." };
  box.innerHTML = `
    <div class="grid grid-cols-[1fr_auto_1fr_auto_1fr] items-center gap-2 text-center">
      <div class="panel py-2"><div class="text-[11px]">Regional band (ML)</div><div class="mt-1">${badge(p.regional_band)}</div></div>
      <div class="text-[16px] text-[var(--muted)]">×</div>
      <div class="panel py-2" data-tip="${esc(sensTip[p.sensitivity])}"><div class="text-[11px]">Your sensitivity</div><div class="mt-1 text-[13px] font-semibold text-[var(--heading)] capitalize">${esc(p.sensitivity)}</div></div>
      <div class="text-[16px] text-[var(--muted)]">→</div>
      <div class="panel py-2"><div class="text-[11px]">Personal level</div><div class="mt-1">${badge(p.personal_level)}</div></div>
    </div>
    ${d.key === "cardio" ? '<p class="text-[11px] mt-2 text-[var(--muted)]">The cardio regional band follows the live AQI category (US EPA), because the cardio model has no predictive skill.</p>' : ""}
    <div class="mt-3 text-[12px] font-semibold text-[var(--heading)]">Rules that fired</div>
    <div class="mt-1 flex flex-wrap gap-1">${p.rules_fired.length ? p.rules_fired.map((r) =>
      `<span class="chip !cursor-help" data-tip="${esc(`${r.sensitivity} sensitivity — ${r.source}`)}">${esc(r.label)} · ${esc(r.sensitivity)}</span>`).join("")
      : '<span class="text-[12px] text-[var(--muted)]">None — normal sensitivity.</span>'}</div>
    <div class="mt-3" id="pa-item"></div>`;
  Components.personalAdvisories(document.getElementById("pa-item"),
    { lang: "en", items: [{ ...p, disease: d.key }], red_flags: p.red_flags }, { large: true });
}

function load() {
  setActiveTab();
  return Page.run(() => apiLoc(`/api/disease/${current}`), (d) => {
    fillHeader(d);
    renderTop(d);
    renderBreakdown(d);
    renderReliability(d);
    renderSpark(d);
    renderShap(d);
    renderAdvisory(d);
    renderPersonal(d);
  });
}

Components.tabs("disease-tabs", (key) => {
  current = key;
  history.replaceState(null, "", `${location.pathname}${location.search}#${key}`);
  load();
});
load();
