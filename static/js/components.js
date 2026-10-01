/* Shared renderers used across pages (bar list, signed bars, advisories, tabs, icons). */
const DISEASE_ICONS = {
  respiratory: `<svg viewBox="0 0 40 40" class="w-[38px] h-[38px]"><path d="M20 6v13" stroke="#E5484D" stroke-width="2.4" stroke-linecap="round"/><path d="M18.5 16c-3-4-9-5-11.5 2-2 6-1.5 13 2.5 14.5 3.5 1.2 7-.5 8-4.5z" fill="#F7787C"/><path d="M21.5 16c3-4 9-5 11.5 2 2 6 1.5 13-2.5 14.5-3.5 1.2-7-.5-8-4.5z" fill="#E5484D"/><path d="M20 19l-4 4M20 19l4 4" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  vector: `<svg viewBox="0 0 40 40" class="w-[38px] h-[38px]"><ellipse cx="20" cy="21" rx="3" ry="8" fill="#1F8A5B"/><circle cx="20" cy="11" r="3" fill="#1F8A5B"/><path d="M20 9l-3-5M20 9l3-5" stroke="#1F8A5B" stroke-width="1.4"/><path d="M18 18C12 12 6 13 5 16c4 1 9 2 13 4zM22 18c6-6 12-5 13-2-4 1-9 2-13 4z" fill="#8FD9B6"/><path d="M18 24l-9 5M18 27l-7 8M22 24l9 5M22 27l7 8M18 21l-10 0M22 21h10" stroke="#1F8A5B" stroke-width="1.3" stroke-linecap="round"/></svg>`,
  heat: `<svg viewBox="0 0 40 40" class="w-[38px] h-[38px]"><circle cx="15" cy="16" r="6" fill="#F5A524"/><g stroke="#F5A524" stroke-width="2" stroke-linecap="round"><path d="M15 5v2.5M15 24.5V27M4 16h2.5M5.5 8.5l1.8 1.8M5.5 23.5l1.8-1.8M22.7 8.5l-1.8 1.8"/></g><rect x="24" y="6" width="6" height="20" rx="3" fill="#FDE7E8" stroke="#E5484D" stroke-width="1.6"/><circle cx="27" cy="29" r="5" fill="#E5484D"/><rect x="26" y="14" width="2" height="14" fill="#E5484D"/></svg>`,
  waterborne: `<svg viewBox="0 0 40 40" class="w-[38px] h-[38px]"><defs><linearGradient id="dg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6FD0F2"/><stop offset="1" stop-color="#1597BB"/></linearGradient></defs><path d="M20 5c5 7 10 12.5 10 18.5a10 10 0 0 1-20 0C10 17.5 15 12 20 5z" fill="url(#dg)"/><path d="M15 24a5 5 0 0 0 4 5" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" opacity=".8"/></svg>`,
  cardio: `<svg viewBox="0 0 40 40" class="w-[38px] h-[38px]"><path d="M20 33S5 24.5 5 14.5A7.5 7.5 0 0 1 20 11a7.5 7.5 0 0 1 15 3.5C35 24.5 20 33 20 33z" fill="#F26B5B"/><path d="M7 19h7l2.5-5 4 10 3-7 2 2h7.5" stroke="#fff" stroke-width="2" fill="none" stroke-linejoin="round" stroke-linecap="round"/></svg>`,
};

const DISEASE_LABELS = { respiratory: "Respiratory", vector: "Vector-borne", heat: "Heat-related",
                         waterborne: "Waterborne", cardio: "Cardiovascular", overall: "Overall" };

const FACTOR_COLORS = ["#EF4C55", "#F47C5E", "#F0A93B", "#1BA3C6", "#3DB9D8", "#6CCFE4"];

const Components = {
  /**
   * Horizontal bar list (track + fill). items: [{label, value 0–100, color?, right?, tip?, icon?}]
   */
  barList(el, items) {
    el.classList.add("bar-list");
    el.innerHTML = items.map((it, i) => {
      const color = it.color || FACTOR_COLORS[i] || FACTOR_COLORS.at(-1);
      return `
      <div class="bar-row">
        <span class="text-[var(--heading)] truncate tip-left" ${it.tip ? `data-tip="${esc(it.tip)}"` : ""}>${esc(it.label)}${it.icon ? ` <i data-lucide="${it.icon}" class="inline w-3 h-3 text-[var(--muted)] align-[-2px]"></i>` : ""}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(it.value, 3)}%;background:${color}"></div></div>
        <span class="text-right text-[var(--heading)]">${it.valueText ?? it.value}</span>
        <span class="text-[10px] flex items-center gap-1 whitespace-nowrap" style="color:${color}">${it.right ?? ""}</span>
      </div>`;
    }).join("");
  },

  /** Signed horizontal bars (SHAP waterfall style): positive = red, negative = green. */
  signedBars(canvasId, rows, unitLabel) {
    const c = Chart.getChart(canvasId);
    if (c) c.destroy();
    const labels = rows.map((r) => r.label + (r.autoregressive ? " ⟲" : ""));
    return new Chart(document.getElementById(canvasId), {
      type: "bar",
      data: { labels, datasets: [{ data: rows.map((r) => r.shap),
        backgroundColor: rows.map((r) => (r.shap >= 0 ? RISK.High.c : RISK.Low.c)),
        borderRadius: 4, barThickness: 16 }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: {
            label: (ctx) => {
              const r = rows[ctx.dataIndex];
              const lines = [`${r.shap >= 0 ? "+" : ""}${r.shap.toFixed(3)} (${unitLabel})`];
              if (r.autoregressive) lines.push("Past values of this indicator (autoregressive feature)");
              return lines;
            },
          } },
        },
        scales: {
          x: { grid: { color: cssVar("--grid") }, ticks: { color: cssVar("--muted"), font: { size: 10 } },
               title: { display: true, text: unitLabel, color: cssVar("--muted"), font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: cssVar("--heading"), font: { size: 11 } } },
        },
      },
    });
  },

  /** Advisory blocks (shared by Dashboard, My Risk and the map drawer). */
  advisories(el, list) {
    const toneColor = { high: RISK.High.c, moderate: RISK.Moderate.c, low: RISK.Low.c };
    el.innerHTML = list.map((a) => `
      <div class="adv tone-${a.tone} flex gap-3">
        <div class="w-[30px] shrink-0 pt-[2px] grid justify-items-center"><i data-lucide="${a.icon}" class="w-[22px] h-[22px]" style="color:${toneColor[a.tone]}"></i></div>
        <div class="min-w-0">
          <div class="adv-title">${esc(a.title)}</div>
          <div class="text-[10px] mt-[1px]">${esc(a.reason)}</div>
          <ul>${a.bullets.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
        </div>
      </div>`).join("");
  },

  /**
   * Personal advisory items (rule-based): level badge, actions, time window,
   * a "Why?" expander with the rules that fired and their sources, then red flags.
   */
  personalAdvisories(el, pa, { large = false } = {}) {
    const icon = { respiratory: "wind", heat: "sun", cardio: "heart-pulse", vector: "bug",
                   waterborne: "droplet", general: "circle-check" };
    const tone = (lvl) => ({ "Very High": "high", High: "high", Moderate: "moderate" }[lvl] || "low");
    const bn = pa.lang === "bn";
    const L = bn ? { why: "কেন?", when: "কখন", rules: "যে নিয়ম প্রযোজ্য", src: "উৎস", you: "আপনার জন্য", flags: "এই লক্ষণ দেখা দিলে" }
                 : { why: "Why?", when: "When", rules: "Rules that fired", src: "Sources", you: "Personalized for you", flags: "Get help now if you notice" };
    const items = pa.items.map((a) => `
      <div class="adv ${large ? "adv-lg" : ""} tone-${tone(a.personal_level)} flex gap-3">
        <div class="w-[30px] shrink-0 pt-[2px] grid justify-items-center"><i data-lucide="${icon[a.disease] || "info"}" class="w-[22px] h-[22px]" style="color:${RISK[a.personal_level].c}"></i></div>
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="adv-title">${esc(a.title)}</span>${badge(a.personal_level, a.personal_level_text)}
            <span class="chip chip-personal !cursor-default">${L.you}</span>
          </div>
          <ul>${a.actions.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
          ${a.when ? `<div class="adv-when"><i data-lucide="clock" class="inline w-3 h-3 align-[-2px]"></i> ${esc(a.when)}</div>` : ""}
          <details class="adv-why">
            <summary>${L.why}</summary>
            <p>${esc(a.why)}</p>
            ${a.overrides?.length ? `<p>${a.overrides.map(esc).join(" ")}</p>` : ""}
            ${a.rules_fired?.length ? `<div class="mt-1"><b>${L.rules}:</b> ${a.rules_fired.map((r) =>
              `<span class="chip !cursor-help" data-tip="${esc(`${r.sensitivity} sensitivity — ${r.source}`)}">${esc(r.label)}</span>`).join(" ")}</div>` : ""}
            <div class="mt-1"><b>${L.src}:</b> ${a.sources.map((s) =>
              `<a href="${esc(s.url)}" target="_blank" rel="noopener" data-tip="${esc(s.label)}">${esc(s.label)}</a>`).join(" · ")}</div>
          </details>
        </div>
      </div>`).join("");
    const flags = pa.red_flags?.length ? `
      <div class="red-flags">
        <div class="flex items-center gap-2 font-semibold"><i data-lucide="siren" class="w-4 h-4"></i>${L.flags}</div>
        <ul>${pa.red_flags.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
      </div>` : "";
    el.innerHTML = items + flags;
  },

  /** Wire pill tabs; onChange(key). */
  tabs(containerId, onChange) {
    const box = document.getElementById(containerId);
    box.addEventListener("click", (ev) => {
      const b = ev.target.closest(".tab");
      if (!b) return;
      box.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
      onChange(b.dataset.key);
    });
  },

  /** Background zones for Low/Moderate/High on a 0–100 score axis (Chart.js plugin). */
  riskBands: {
    id: "riskBands",
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea: a, scales: { y } } = chart;
      if (!y) return;
      ctx.save();
      [[0, 40, RISK.Low.c], [40, 65, RISK.Moderate.c], [65, 100, RISK.High.c]].forEach(([lo, hi, c]) => {
        const top = y.getPixelForValue(Math.min(hi, y.max));
        const bot = y.getPixelForValue(Math.max(lo, y.min));
        if (bot <= top) return;
        ctx.fillStyle = c + "14";
        ctx.fillRect(a.left, top, a.right - a.left, bot - top);
      });
      ctx.restore();
    },
  },
};
