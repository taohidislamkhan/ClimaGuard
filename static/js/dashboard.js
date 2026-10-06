/* Dashboard page: fetch /api/dashboard and render every card from it. */
const state = { data: null, trendKey: "overall", profile: null };

const ENV_TILES = [
  { key: "temperature", label: "Temperature", icon: "thermometer", unit: "°C", d: 0 },
  { key: "humidity",    label: "Humidity",    icon: "droplets",    unit: "%",  d: 0 },
  { key: "aqi",         label: "AQI",         icon: "leaf",        unit: "",   d: 0 },
  { key: "pm25",        label: "PM2.5",       icon: "circle-gauge", unit: " µg/m³", d: 0 },
  { key: "rainfall",    label: "Rainfall",    icon: "cloud-rain",  unit: " mm", d: 1 },
  { key: "wind_speed",  label: "Wind Speed",  icon: "wind",        unit: " km/h", d: 0 },
];

function arrow(dir) {
  const name = { up: "arrow-up", down: "arrow-down", flat: "minus" }[dir];
  return name ? `<i data-lucide="${name}" class="trend-arrow text-[#1F6FEB]"></i>` : "";
}

/* Score change in points, e.g. "↑ 6 pts vs last week (dataset)". */
function changeHtml(points, label, tip) {
  if (points === null || points === undefined) {
    return `<span class="text-[var(--muted)]" data-tip="${esc(tip)}">– ${esc(label)}</span>`;
  }
  const n = Math.round(points);
  if (n === 0) {
    return `<span class="inline-flex items-center gap-1" data-tip="${esc(tip)}"><i data-lucide="minus" class="w-3 h-3"></i>0 pts ${esc(label)}</span>`;
  }
  const up = n > 0;
  return `<span data-tip="${esc(tip)}"><i data-lucide="${up ? "arrow-up" : "arrow-down"}" class="inline w-3 h-3 align-[-2px] ${up ? "up" : "down"}" style="stroke-width:2.6"></i>
      <span class="${up ? "up" : "down"} whitespace-nowrap">${Math.abs(n)} pts</span> ${esc(label)}</span>`;
}

function prevTip(d) {
  const c = d.changes;
  if (!c.previous_date) return "No earlier snapshot yet.";
  if (c.previous_source === "backfill") {
    return `Demo: compared with the backfilled dataset week of ${c.previous_date} (model run on the last dataset weeks). Real day-to-day history builds up as the dashboard is used.`;
  }
  return `Compared with the snapshot from ${c.previous_date}.`;
}

/* ---------------- renderers ---------------- */
function renderHero(d) {
  const o = d.overall;
  document.getElementById("overall-score").textContent = o.score;
  Charts.gauge("gauge", o.score, o.level);
  const badge = document.getElementById("overall-badge");
  badge.className = `badge !h-[24px] !px-[11px] !text-[12px] lvl-${o.level}`;
  badge.textContent = `${o.level} risk`;
  const p = o.probabilities;
  badge.setAttribute("data-tip",
    `${o.model} classifier: P(Low) ${(p.Low * 100).toFixed(0)}%, P(Moderate) ${(p.Moderate * 100).toFixed(0)}%, P(High) ${(p.High * 100).toFixed(0)}%. Score = ${o.formula}. Badge = most likely class.`);
  badge.classList.add("tip-left");
  const demo = o.demo_change ? " " + chip("Demo Data", prevTip(d)) : "";
  document.getElementById("overall-change").innerHTML =
    changeHtml(o.change_points, o.compare_label, prevTip(d)) + demo;
  document.getElementById("overall-explain").textContent = d.live
    ? `Based on current weather and air quality in ${d.location.name}.${d.profile ? " Your profile adjusts the disease scores." : ""}`
    : `Regional estimate from the latest dataset week (${d.data_date}); live weather is unavailable.`;
  document.getElementById("disclaimer").textContent = d.disclaimer;
  document.getElementById("last-updated").textContent = fmt.time(d.updated_at);
}

function renderDiseases(d) {
  const tip = prevTip(d);
  document.getElementById("disease-cards").innerHTML = d.diseases.map((x) => {
    const adj = x.adjustment.points
      ? `<span class="chip ml-1 tip-left" data-tip="Model score ${x.model_score} + profile adjustment ${x.adjustment.points > 0 ? "+" : ""}${x.adjustment.points} (${esc(x.adjustment.reasons.join(", "))}). Rule-based, applied after the model; never fed into it.">${x.adjustment.points > 0 ? "+" : ""}${x.adjustment.points} profile</span>`
      : "";
    const info = `${x.model} predicts ${x.prediction} (${x.target}), which is the ${x.model_score}th percentile of training weeks (2015–2022). Bands: <40 Low, 40–64 Moderate, ≥65 High.${x.note ? " " + x.note : ""}`;
    return `
    <div class="card relative min-h-[135px] px-[14px] pt-[13px] pb-[10px] flex flex-col lvl-${x.level}">
      <div class="absolute inset-x-0 bottom-0 h-[34px] rounded-b-[12px]" style="background:linear-gradient(180deg, transparent, color-mix(in srgb, var(--cbg) 60%, transparent))"></div>
      <div class="flex gap-[12px] 2xl:gap-[18px] relative">
        <div class="pt-[2px] shrink-0">${DISEASE_ICONS[x.key]}</div>
        <div class="min-w-0 flex-1">
          <div class="text-[12px] 2xl:text-[12.5px] font-semibold text-[var(--heading)] leading-snug">${esc(x.label)} <span class="whitespace-nowrap">Risk<span class="tip-left inline-block align-[-1px] ml-1" data-tip="${esc(info)}"><i data-lucide="info" class="w-3 h-3 text-[var(--muted)]"></i></span></span></div>
          <div class="text-[20px] font-bold text-[var(--heading)] mt-[2px] leading-tight">${x.score}</div>
          <div class="mt-[4px] flex flex-wrap items-center gap-y-1"><span class="badge lvl-${x.level}">${x.level}</span>${adj}</div>
        </div>
      </div>
      <div class="relative mt-auto pt-[6px] flex items-end justify-between gap-2">
        <div class="text-[9.5px] leading-tight min-w-0">${changeHtml(x.change_points, x.compare_label, tip)}</div>
        <div class="w-[52px] h-[30px] shrink-0"><canvas id="spark-${x.key}"></canvas></div>
      </div>
    </div>`;
  }).join("");
  d.diseases.forEach((x) => Charts.sparkline(`spark-${x.key}`, x.sparkline, x.level));
}

function renderFactors(d) {
  document.getElementById("factors").innerHTML = d.shap_factors.map((f, i) => {
    const color = FACTOR_COLORS[i] || FACTOR_COLORS.at(-1);
    const ar = f.autoregressive ? " Past values of this indicator (autoregressive feature)." : "";
    const dirTip = `${f.label} ${f.direction} today's High-risk estimate (SHAP ${f.shap > 0 ? "+" : ""}${f.shap.toFixed(2)} on the High-class log-odds).${ar} Features: ${f.features.join(", ")}`;
    return `
    <div class="bar-row">
      <span class="text-[var(--heading)] truncate tip-left" data-tip="${esc(dirTip)}">${esc(f.label)}${f.autoregressive ? ' <i data-lucide="history" class="inline w-3 h-3 text-[var(--muted)] align-[-2px]"></i>' : ""}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(f.value, 3)}%;background:${color}"></div></div>
      <span class="text-right text-[var(--heading)]">${f.value}</span>
      <span class="text-[10px] flex items-center gap-1 whitespace-nowrap" style="color:${color}">
        <i data-lucide="${f.direction === "raises" ? "trending-up" : "trending-down"}" class="w-3 h-3"></i>${f.level}<span class="lvl-word -ml-[2px]">contribution</span></span>
    </div>`;
  }).join("");
}

function renderEnvironment(d) {
  const e = d.environment;
  document.getElementById("env-loc").innerHTML =
    `${esc(e.location)}, Bangladesh <span class="text-[var(--muted)] ml-1" data-tip="Temperature, rainfall, PM2.5 and AQI are model inputs (the model uses their weekly aggregates). Humidity, wind speed and UV index are live display-only values — they are NOT used by the model. Arrows compare with 24 hours earlier. Source: Open-Meteo.">· ${e.live ? "live" : "dataset"} <i data-lucide="info" class="inline w-3 h-3"></i></span>`;
  const aqiBad = (e.values.aqi ?? 0) > 100;
  document.getElementById("env-grid").innerHTML = ENV_TILES.map((t) => {
    const v = e.values[t.key];
    const red = t.key === "aqi" && aqiBad;
    const disp = e.display_only.includes(t.key);
    return `
    <div class="env-tile pl-[8px]" ${disp ? `data-tip="Display only — not a model input."` : ""}>
      <div class="flex items-end gap-[4px]">
        <i data-lucide="${t.icon}" class="env-icon mb-[1px] ${red ? "text-[#E5484D]" : "text-[#1F6FEB]"}"></i>
        <div class="flex flex-col">
          <span class="h-[11px]">${arrow(e.trend[t.key])}</span>
          <div class="env-val ${red ? "!text-[#E5484D] !text-[18px]" : ""}">${v === null || v === undefined ? "–" : fmt.num(v, t.d)}<small>${v === null || v === undefined ? "" : t.unit}</small></div>
        </div>
      </div>
      <div class="env-lbl mt-[4px]">${t.label}${disp ? '<sup class="text-[var(--muted)]">*</sup>' : ""}</div>
    </div>`;
  }).join("");
  const uv = e.values.uv_index;
  document.getElementById("env-uv").innerHTML = `
    <div class="env-tile pl-[8px]" data-tip="Display only — not a model input. Today's maximum UV index.">
      <div class="flex items-end gap-[4px]"><i data-lucide="sun" class="env-icon mb-[1px] text-[#F5A524]"></i>
        <div class="flex flex-col"><span class="h-[11px]">${arrow(e.trend.uv_index)}</span>
        <div class="env-val">${uv === null || uv === undefined ? "–" : fmt.num(uv, 0)}</div></div></div>
      <div class="env-lbl mt-[4px]">UV Index<sup class="text-[var(--muted)]">*</sup></div>
    </div>`;
  const cat = e.aqi_category;
  const lvl = cat.level || "Moderate";
  const box = document.getElementById("aqi-callout");
  box.style.background = `color-mix(in srgb, var(--${lvl.toLowerCase()}-bg) 70%, transparent)`;
  box.style.border = `1px solid var(--${lvl.toLowerCase()}-bg)`;
  box.innerHTML = `<i data-lucide="${lvl === "Low" ? "circle-check" : "circle-alert"}" class="w-4 h-4 shrink-0" style="color:${RISK[lvl].c}"></i>
    <div class="leading-tight"><div class="text-[var(--heading)]">Air Quality</div><div style="color:${lvl === "Moderate" ? "#C98200" : RISK[lvl].c}">${esc(cat.label)}</div></div>`;
}

function renderTrend(d) {
  const t = d.trend;
  document.getElementById("trend-chip").innerHTML = t.demo
    ? chip("Demo Data", "Earlier points are backfilled: the models were run on the last dataset weeks (2025), so they are weekly, not daily. They are replaced by real daily snapshots as the dashboard is used.")
    : "";
  const titles = t.labels.map((l, i) => `${l} · ${t.sources[i] === "live" ? "live snapshot" : "backfilled dataset week"}`);
  Charts.trend("trend-chart", t.labels, t[state.trendKey], titles);
}

function renderAdvisories(d) {
  Components.advisories(document.getElementById("advisories"), d.advisories);
  document.getElementById("adv-footer").innerHTML =
    `<b class="font-medium text-[var(--body)]">${esc(d.disclaimer)}</b> ${esc(d.advisory_footer)}`;
  loadPersonal();
}

/* Personal advisories replace the regional ones once a profile is saved. */
async function loadPersonal() {
  if (!USER) return;                                  // guests keep the regional advisories
  let pa;
  try { pa = await apiLoc(`/api/advisory/personal?lang=${Lang.get()}`); } catch (e) { console.error(e); return; }
  if (!pa.personalized) return;                       // keep the regional advisories
  Components.personalAdvisories(document.getElementById("advisories"), pa);
  document.getElementById("adv-chip").innerHTML = chip("Personalized for you",
    "Rule-based: your regional (ML) risk band × your sensitivity from the My Health profile, plus live air and heat checks. " +
    "Your profile never enters the ML models. Open \"Why?\" on an item for the rules and their public-health sources.");
  document.getElementById("adv-footer").innerHTML = `<b class="font-medium text-[var(--body)]">${esc(pa.disclaimer)}</b>`;
  const tg = document.getElementById("adv-lang");
  tg.classList.remove("hidden");
  tg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.lang === pa.lang));
  lucide.createIcons();
}

document.getElementById("adv-lang").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (b && b.dataset.lang !== Lang.get()) { Lang.set(b.dataset.lang); loadPersonal(); }
});

function renderChanges(d) {
  const c = d.changes;
  document.getElementById("changes-chip").innerHTML = c.demo ? chip("Demo Data", prevTip(d)) : "";
  const since = { "vs yesterday": "yesterday", "vs last snapshot": `the last snapshot (${c.previous_date})` }[c.compare_label]
    || `the last dataset week (${c.previous_date})`;
  let text = "No earlier snapshot to compare with yet.";
  if (c.points !== null) {
    const n = Math.abs(c.points).toFixed(0);
    text = c.points === 0 ? `Your risk is unchanged compared with ${since}.`
      : `Your risk ${c.points > 0 ? "increased" : "decreased"} by <b class="${c.points > 0 ? "up" : "down"}">${n}</b> points compared with ${esc(since)}.`;
  }
  document.getElementById("changes-text").innerHTML = text;
  document.getElementById("contributors").innerHTML = c.contributors.map((x) => {
    const p = x.change_pct;
    const val = p === null ? "–" : `<i data-lucide="${p >= 0 ? "arrow-up" : "arrow-down"}" class="w-3 h-3" style="stroke-width:2.6"></i>${Math.abs(p).toFixed(0)}%`;
    return `<div class="flex justify-between text-[11px] pr-2" data-tip="Change in this week's ${esc(x.label)} (a model input) versus the previous snapshot.">
      <span class="text-[var(--heading)]">${esc(x.label)}</span>
      <span class="flex items-center gap-1 font-medium ${p === null ? "" : p >= 0 ? "up" : "down"}">${val}</span></div>`;
  }).join("");
}

function renderMap(d) {
  RiskMap.render("mini-map", d.map, { radius: 5.5 });
  document.getElementById("map-legend").innerHTML = d.map.map((p) => `
    <div class="grid items-center gap-x-1" style="grid-template-columns: 10px 14px minmax(0,1fr) auto;">
      <span class="text-[var(--muted)]">•</span>
      <span class="w-[9px] h-[9px] rounded-full" style="background:${RISK[p.level].c}"></span>
      <span class="text-[var(--heading)]">${esc(p.name)}</span>
      <span style="color:${RISK[p.level].c}" data-tip="Overall score ${p.score}/100">${p.level}</span>
    </div>`).join("");
}

function renderProfile(p) {
  const name = USER ? USER.name : "";
  document.getElementById("greeting").textContent = `${greetingFor()}${name ? ", " + name : ""} 👋`;
  const strip = document.getElementById("profile-strip");
  if (!strip) return;                                 // guest: the log-in card is shown instead
  if (!p) {
    strip.innerHTML = `<div class="col-span-6 text-[12px] py-[6px]">No profile yet. Add your age, conditions and exposure to get personalized scores and advice.</div>`;
    return;
  }
  const items = [
    ["calendar-days", "Age", p.age],
    ["map-pin", "Location", p.location],
    ["calculator", "BMI", p.bmi],
    ["person-standing", "Activity", p.activity],
    ["sun", "Outdoor Exposure", p.outdoor_exposure],
    ["cigarette-off", "Smoking", p.smoking ? "Yes" : "No"],
  ];
  const colors = { "Outdoor Exposure": "#F5A524", Smoking: "#E5484D" };
  document.getElementById("profile-strip").innerHTML = items.map(([icon, label, v]) => `
    <div class="flex items-center gap-[14px]">
      <i data-lucide="${icon}" class="w-[20px] h-[20px]" style="color:${colors[label] || "#1F6FEB"}"></i>
      <div class="leading-tight"><div class="text-[10.5px]">${label}</div><div class="text-[11.5px] font-semibold text-[var(--heading)] mt-[2px]">${esc(v ?? "–")}</div></div>
    </div>`).join("");
}

function render(d) {
  state.data = d;
  state.profile = d.profile;
  fillHeader(d);
  renderHero(d);
  renderDiseases(d);
  renderFactors(d);
  renderEnvironment(d);
  renderTrend(d);
  renderAdvisories(d);
  renderChanges(d);
  renderMap(d);
  renderProfile(d.profile);
}

/* ---------------- interactions ---------------- */
function load(recalc = false) {
  const btn = document.getElementById("recalc");     // signed-in users only
  btn?.classList.add("loading");
  if (btn) btn.disabled = true;
  return Page.run(
    () => (recalc ? apiLoc("/api/recalculate", { method: "POST", body: "{}" }) : apiLoc("/api/dashboard")),
    render,
  ).finally(() => {
    btn?.classList.remove("loading");
    if (btn) btn.disabled = false;
  });
}

document.getElementById("recalc")?.addEventListener("click", () => load(true));
// The user's "data refresh interval" setting reloads the dashboard data.
if (Page.autoRefreshMinutes) setInterval(() => load(false), Page.autoRefreshMinutes * 60 * 1000);

Components.tabs("trend-tabs", (key) => {
  state.trendKey = key;
  if (state.data) renderTrend(state.data);
});

load(false);
