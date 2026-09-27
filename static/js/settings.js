/* Settings: each change is saved immediately via POST /api/settings. */
let current = null;

function setStatus(text, kind = "") {
  const el = document.getElementById("st-status");
  el.textContent = text;
  el.style.color = kind === "error" ? "var(--high)" : kind === "ok" ? "var(--low)" : "var(--muted)";
}

function show(s) {
  current = s;
  document.querySelectorAll("[data-setting]").forEach((seg) => {
    const val = String(s[seg.dataset.setting]);
    seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.value === val));
  });
  document.getElementById("st-default-loc").value = s.default_location;
  document.querySelectorAll("[data-notify]").forEach((c) => { c.checked = !!s.notifications[c.dataset.notify]; });
}

async function save(patch) {
  setStatus("Saving…");
  try {
    const s = await api("/api/settings", { method: "POST", body: JSON.stringify(patch) });
    Object.assign(SETTINGS, s);
    show(s);
    document.documentElement.dataset.theme = s.theme;      // apply theme live
    setStatus("Saved.", "ok");
  } catch (e) {
    show(current);
    setStatus(e.message, "error");
  }
}

document.querySelectorAll("[data-setting]").forEach((seg) => {
  seg.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    const key = seg.dataset.setting;
    save({ [key]: key === "refresh_minutes" ? Number(b.dataset.value) : b.dataset.value });
  });
});
document.getElementById("st-default-loc").addEventListener("change", (ev) => save({ default_location: ev.target.value }));
document.querySelectorAll("[data-notify]").forEach((c) =>
  c.addEventListener("change", () => save({ notifications: { [c.dataset.notify]: c.checked } })));

function renderAbout(a) {
  const ds = a.dataset;
  const rows = [
    ["App version", a.app_version],
    ["Dataset", `${ds.file}`],
    ["Coverage", `${ds.rows.toLocaleString()} weekly rows · ${ds.countries} countries · ${ds.start} → ${ds.end}`],
    ["Model files written", `${a.last_training} (Asia/Dhaka, file timestamp)`],
    ["Libraries", Object.entries(a.libraries).map(([k, v]) => `${k} ${v}`).join(" · ")],
    ["Live data", "Open-Meteo forecast, historical-forecast and air-quality APIs"],
  ];
  document.getElementById("about").innerHTML = rows.map(([k, v]) =>
    `<dt class="text-[var(--muted)]">${esc(k)}</dt><dd class="text-[var(--heading)] break-words">${esc(v)}</dd>`).join("");
  document.getElementById("about-models").innerHTML = a.models.map((m) =>
    `<tr><td>${esc(DISEASE_LABELS[m.task] || m.task)}</td><td>${esc(m.model)}</td><td><code class="text-[11px]">${esc(m.file)}</code></td>
     <td>${m.size_kb >= 1024 ? (m.size_kb / 1024).toFixed(1) + " MB" : m.size_kb + " KB"}</td></tr>`).join("");
  document.querySelector("#about-disclaimer span").textContent = a.disclaimer;
}

Page.run(async () => {
  const [s, a] = await Promise.all([api("/api/settings"), api("/api/about")]);
  return { settings: s, about: a };
}, ({ settings, about }) => {
  show(settings);
  renderAbout(about);
});
initHeaderOnly();
