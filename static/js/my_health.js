/* My Health: profile form (saved to SQLite via /api/profile) + live adjustment preview. */
const form = document.getElementById("hp-form");
const F = form.elements;   // F.name would be the <form> name attribute, not the input
const status = document.getElementById("hp-status");
let saved = null;
let bmiMode = "bmi";
let previewTimer = null;

function setBmiMode(mode) {
  bmiMode = mode;
  document.querySelectorAll("#bmi-mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  document.getElementById("bmi-field").classList.toggle("hidden", mode !== "bmi");
  document.querySelectorAll("[data-hw]").forEach((el) => el.classList.toggle("hidden", mode !== "hw"));
}

function fill(p) {
  F.name.value = p.name ?? "";
  F.age.value = p.age ?? "";
  F.location.value = p.location;
  F.bmi.value = p.bmi ?? "";
  F.height_cm.value = p.height_cm ?? "";
  F.weight_kg.value = p.weight_kg ?? "";
  F.activity.value = p.activity;
  F.outdoor_exposure.value = p.outdoor_exposure;
  ["smoking", "asthma", "cardiovascular_disease", "diabetes"].forEach((k) => { F[k].checked = !!p[k]; });
  setBmiMode(p.height_cm && p.weight_kg ? "hw" : "bmi");
  updateAutoBmi();
}

function collect() {
  const n = (v) => (v === "" ? null : Number(v));
  const body = {
    name: F.name.value.trim(), age: n(F.age.value), location: F.location.value,
    activity: F.activity.value, outdoor_exposure: F.outdoor_exposure.value,
    smoking: F.smoking.checked, asthma: F.asthma.checked,
    cardiovascular_disease: F.cardiovascular_disease.checked, diabetes: F.diabetes.checked,
  };
  if (bmiMode === "hw") {
    body.height_cm = n(F.height_cm.value);
    body.weight_kg = n(F.weight_kg.value);
  } else {
    body.height_cm = null;
    body.weight_kg = null;
    body.bmi = n(F.bmi.value);
  }
  return body;
}

function updateAutoBmi() {
  const h = Number(F.height_cm.value), w = Number(F.weight_kg.value);
  document.getElementById("bmi-auto").textContent = h > 0 && w > 0 ? (w / (h / 100) ** 2).toFixed(1) : "–";
}

function setStatus(text, kind = "") {
  status.textContent = text;
  status.style.color = kind === "error" ? "var(--high)" : kind === "ok" ? "var(--low)" : "var(--muted)";
}

function renderPreview(pv) {
  const fx = (effects) => Object.entries(effects)
    .map(([d, p]) => `<span class="chip !cursor-default">${DISEASE_LABELS[d]} +${p}</span>`).join(" ");
  document.getElementById("pv-rules").innerHTML = pv.rules.map((r) => `
    <div class="flex items-center gap-2 text-[12.5px] ${r.fires ? "" : "opacity-55"}">
      <i data-lucide="${r.fires ? "circle-check" : "circle"}" class="w-4 h-4 shrink-0" style="color:${r.fires ? RISK.Low.c : "var(--muted)"}"></i>
      <span class="w-[170px] shrink-0 ${r.fires ? "font-semibold text-[var(--heading)]" : ""}">${esc(r.label)}</span>
      <span class="flex flex-wrap gap-1">${fx(r.effects)}</span>
    </div>`).join("");
  document.getElementById("pv-loc").textContent = Loc.get();
  const level = (v) => (v >= 65 ? "High" : v >= 40 ? "Moderate" : "Low");
  document.getElementById("pv-rows").innerHTML = pv.diseases.map((d) => `
    <tr><td>${esc(d.label)}</td>
      <td>${d.model_score} ${badge(level(d.model_score))}</td>
      <td class="${d.points > 0 ? "up" : ""}">${d.points > 0 ? "+" : ""}${d.points}${d.capped ? ` <span class="text-[11px] text-[var(--muted)]" data-tip="Rules add up to +${d.raw_points}; capped at ±${pv.cap}.">(capped from +${d.raw_points})</span>` : ""}</td>
      <td><b>${d.score}</b> ${badge(level(d.score))}</td></tr>`).join("");
  lucide.createIcons();
}

async function preview() {
  try {
    renderPreview(await apiLoc("/api/profile/preview", { method: "POST", body: JSON.stringify(collect()) }));
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("invalid"));
  } catch (e) {
    setStatus(e.message, "error");
  }
}

function schedulePreview() {
  updateAutoBmi();
  setStatus(saved && JSON.stringify(collect()) !== JSON.stringify(saved) ? "Unsaved changes" : "");
  clearTimeout(previewTimer);
  previewTimer = setTimeout(preview, 250);
}

form.addEventListener("input", schedulePreview);
form.addEventListener("change", schedulePreview);
document.getElementById("bmi-mode").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (b) { setBmiMode(b.dataset.mode); schedulePreview(); }
});
document.getElementById("hp-reset").addEventListener("click", () => { fill(saved); schedulePreview(); setStatus(""); });

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = document.getElementById("hp-save");
  btn.disabled = true;
  setStatus("Saving…");
  try {
    const p = await api("/api/profile", { method: "POST", body: JSON.stringify(collect()) });
    fill(p);
    saved = collect();
    setProfileHeader(p);
    setStatus("Saved.", "ok");
    preview();
  } catch (e) {
    setStatus(e.message, "error");
    const field = { Age: "age", BMI: "bmi", Height: "height_cm", Weight: "weight_kg" }[e.message.split(" ")[0]];
    if (field) F[field].closest(".field").classList.add("invalid");
  } finally {
    btn.disabled = false;
  }
});

Page.run(() => api("/api/profile"), (p) => {
  fill(p);
  saved = collect();
  preview();
});
initHeaderOnly();
