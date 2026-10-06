/* Admin pages: users table (search, pagination, actions), audit log, model & data actions.
   Every state-changing call goes through api() (CSRF header) after a confirm dialog. */

/* ---------------- confirm dialog ---------------- */
const Dialog = {
  el: null,
  build() {
    const el = document.createElement("div");
    el.className = "dialog-backdrop hidden";
    el.setAttribute("role", "dialog");
    el.setAttribute("aria-modal", "true");
    el.setAttribute("aria-labelledby", "dlg-title");
    el.innerHTML = `
      <div class="card dialog">
        <h3 class="card-title text-[16px]" id="dlg-title"></h3>
        <div class="text-[13px] mt-2" id="dlg-body"></div>
        <div class="flex justify-end gap-2 mt-5">
          <button type="button" class="btn-ghost !h-[32px]" data-dlg="cancel">Cancel</button>
          <button type="button" class="btn-primary" data-dlg="ok">Confirm</button>
        </div>
      </div>`;
    document.body.appendChild(el);
    return el;
  },
  /** Resolves true on Confirm, false on Cancel / Esc / backdrop click. */
  ask(title, body, { ok = "Confirm", danger = false, info = false } = {}) {
    this.el = this.el || this.build();
    const el = this.el;
    el.querySelector("#dlg-title").textContent = title;
    el.querySelector("#dlg-body").innerHTML = body;           // callers pass escaped HTML
    const okBtn = el.querySelector('[data-dlg="ok"]');
    const cancel = el.querySelector('[data-dlg="cancel"]');
    okBtn.textContent = ok;
    okBtn.style.background = danger ? "var(--high)" : "";
    cancel.classList.toggle("hidden", info);
    el.classList.remove("hidden");
    okBtn.focus();
    return new Promise((resolve) => {
      const done = (v) => {
        el.classList.add("hidden");
        el.removeEventListener("click", onClick);
        document.removeEventListener("keydown", onKey);
        resolve(v);
      };
      const onClick = (ev) => {
        if (ev.target === el) done(false);
        const b = ev.target.closest("[data-dlg]");
        if (b) done(b.dataset.dlg === "ok");
      };
      const onKey = (ev) => { if (ev.key === "Escape") done(false); };
      el.addEventListener("click", onClick);
      document.addEventListener("keydown", onKey);
    });
  },
};

function setStatus(id, text, kind = "") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.style.color = kind === "error" ? "var(--high)" : kind === "ok" ? "var(--low)" : "var(--muted)";
}

function when(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString("en-GB", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function pager(prefix, page, pages, go) {
  document.getElementById(`${prefix}-page`).textContent = `Page ${page} of ${pages}`;
  const prev = document.getElementById(`${prefix}-prev`), next = document.getElementById(`${prefix}-next`);
  prev.disabled = page <= 1;
  next.disabled = page >= pages;
  prev.onclick = () => go(page - 1);
  next.onclick = () => go(page + 1);
}

/* ---------------- Users ---------------- */
const Users = {
  page: 1, q: "", items: [],
  async load(page = this.page) {
    this.page = page;
    const params = new URLSearchParams({ page, q: this.q });
    try {
      const r = await api(`/api/admin/users?${params}`);
      this.items = r.items;
      this.render(r);
    } catch (e) {
      setStatus("u-status", e.message, "error");
    }
  },
  statusCell(u) {
    if (!u.is_active) return `<span class="status-dot" style="background:var(--muted)"></span>Deactivated`;
    if (u.locked) return `<span class="status-dot" style="background:var(--high)"></span>Locked`;
    if (u.must_change_password) return `<span class="status-dot" style="background:var(--moderate)"></span>Temp. password`;
    return `<span class="status-dot" style="background:var(--low)"></span>Active`;
  },
  actions(u) {
    const b = (act, icon, label, tip, dis = false) =>
      `<button type="button" class="btn-ghost !h-[28px] !px-2" data-act="${act}" data-id="${u.id}" data-tip="${esc(tip)}" aria-label="${esc(label)}" ${dis ? "disabled style='opacity:.4'" : ""}><i data-lucide="${icon}"></i></button>`;
    const self = u.is_self;
    return [
      u.role === "admin"
        ? b("demote", "shield-minus", "Demote to user", self ? "You cannot change your own role" : "Demote to user", self)
        : b("promote", "shield-plus", "Promote to admin", "Promote to admin"),
      u.is_active
        ? b("deactivate", "user-round-x", "Deactivate", self ? "You cannot deactivate yourself" : "Deactivate", self)
        : b("activate", "user-round-check", "Activate", "Activate"),
      b("reset", "key-round", "Reset password", self ? "Use the Account page for your own password" : "Set a temporary password", self),
      u.locked ? b("unlock", "lock-open", "Unlock", "Unlock now") : "",
    ].join(" ");
  },
  render(r) {
    document.getElementById("u-count").textContent = `${r.total} account${r.total === 1 ? "" : "s"}`;
    const rows = document.getElementById("u-rows");
    rows.innerHTML = r.items.length ? r.items.map((u) => `
      <tr>
        <td class="font-medium">${esc(u.name)}${u.is_self ? ' <span class="chip !cursor-default">you</span>' : ""}</td>
        <td>${esc(u.email)}</td>
        <td><span class="role-pill ${u.role === "admin" ? "" : "role-user"}">${u.role === "admin" ? "Admin" : "User"}</span></td>
        <td class="whitespace-nowrap">${this.statusCell(u)}</td>
        <td class="text-[12px] whitespace-nowrap">${when(u.created_at)}</td>
        <td class="text-[12px] whitespace-nowrap">${when(u.last_login_at)}</td>
        <td class="text-right whitespace-nowrap">${this.actions(u)}</td>
      </tr>`).join("") : `<tr><td colspan="7"><div class="empty-state">No accounts match.</div></td></tr>`;
    pager("u", r.page, r.pages, (p) => this.load(p));
    lucide.createIcons();
  },
  async act(act, id) {
    const u = this.items.find((x) => x.id === id);
    if (!u) return;
    const who = `<b>${esc(u.name)}</b> (${esc(u.email)})`;
    const spec = {
      promote:    ["Promote to admin?", `${who} will get access to the admin panel (account metadata, audit log, model actions). Admins still cannot see health data.`, "role", { role: "admin" }],
      demote:     ["Demote to user?", `${who} will lose access to the admin panel.`, "role", { role: "user" }],
      deactivate: ["Deactivate account?", `${who} will be signed out everywhere and cannot log in until reactivated.`, "status", { active: false }],
      activate:   ["Activate account?", `${who} will be able to log in again.`, "status", { active: true }],
      reset:      ["Reset password?", `${who} gets a temporary password and must choose a new one at the next login. Their current sessions end.`, "reset-password", {}],
      unlock:     ["Unlock account?", `${who} can try to log in again right away.`, "unlock", {}],
    }[act];
    const danger = ["deactivate", "reset", "demote"].includes(act);
    if (!(await Dialog.ask(spec[0], spec[1], { ok: spec[0].replace("?", ""), danger }))) return;
    setStatus("u-status", "Saving…");
    try {
      const r = await api(`/api/admin/users/${id}/${spec[2]}`, { method: "POST", body: JSON.stringify(spec[3]) });
      setStatus("u-status", "Done. Logged in the audit log.", "ok");
      if (r.temp_password) {
        await Dialog.ask("Temporary password", `Give this password to ${who} through a private channel. It is shown only once and is not stored anywhere in plain text.
          <div class="panel mono text-[16px] text-[var(--heading)] p-3 mt-3 select-all text-center">${esc(r.temp_password)}</div>`, { ok: "Done", info: true });
      }
      this.load();
    } catch (e) {
      setStatus("u-status", e.message, "error");
    }
  },
  init() {
    const search = document.getElementById("u-search");
    let t = null;
    search.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { this.q = search.value.trim(); this.load(1); }, 250); });
    document.getElementById("u-rows").addEventListener("click", (ev) => {
      const b = ev.target.closest("button[data-act]");
      if (b && !b.disabled) this.act(b.dataset.act, Number(b.dataset.id));
    });
    this.load(1);
  },
};

/* ---------------- Audit log ---------------- */
const Audit = {
  page: 1,
  params(page) {
    const f = new FormData(document.getElementById("a-filters"));
    const p = new URLSearchParams();
    for (const [k, v] of f.entries()) if (v) p.set(k, v);
    if (page) p.set("page", page);
    return p;
  },
  async load(page = 1) {
    this.page = page;
    document.getElementById("a-csv").href = `/api/admin/audit.csv?${this.params()}`;
    try {
      const r = await api(`/api/admin/audit?${this.params(page)}`);
      document.getElementById("a-count").textContent = `${r.total} entr${r.total === 1 ? "y" : "ies"}`;
      document.getElementById("a-rows").innerHTML = r.items.length ? r.items.map((x) => `
        <tr><td class="text-[12px] whitespace-nowrap">${esc(x.timestamp.replace("T", " ").replace("Z", ""))}</td>
          <td><span class="chip !cursor-default ${x.action === "login_fail" ? "!bg-[var(--high-bg)] !text-[var(--high)]" : ""}">${esc(x.action)}</span></td>
          <td>${esc(x.actor || (x.actor_user_id ? `user:${x.actor_user_id} (deleted)` : "–"))}</td>
          <td class="text-[12px]">${esc(x.target || "–")}</td>
          <td class="text-[12px] mono">${esc(x.ip || "–")}</td></tr>`).join("")
        : `<tr><td colspan="5"><div class="empty-state">No entries match.</div></td></tr>`;
      pager("a", r.page, r.pages, (p) => this.load(p));
    } catch (e) {
      document.getElementById("a-rows").innerHTML = `<tr><td colspan="5"><div class="empty-state" style="color:var(--high)">${esc(e.message)}</div></td></tr>`;
    }
  },
  init() {
    document.getElementById("a-filters").addEventListener("submit", (ev) => { ev.preventDefault(); this.load(1); });
    this.load(1);
  },
};

/* ---------------- Model & Data actions ---------------- */
function initModelActions() {
  const bind = (id, url, title, okText) => {
    const btn = document.getElementById(id);
    btn.addEventListener("click", async () => {
      if (!(await Dialog.ask(title, esc(btn.dataset.confirm)))) return;
      btn.disabled = true;
      btn.classList.add("loading");
      setStatus("m-status", "Working…");
      try {
        await api(url, { method: "POST", body: "{}" });
        setStatus("m-status", okText, "ok");
      } catch (e) {
        setStatus("m-status", e.message, "error");
      } finally {
        btn.disabled = false;
        btn.classList.remove("loading");
      }
    });
  };
  bind("m-recalc", "/api/admin/recalculate", "Recalculate all divisions?", "All eight divisions recalculated.");
  bind("m-clear", "/api/admin/clear-cache", "Clear the weather cache?", "Cache cleared. The next page load fetches fresh data.");
}

if (document.getElementById("u-rows")) Users.init();
if (document.getElementById("a-rows")) Audit.init();
if (document.getElementById("m-recalc")) initModelActions();
initHeaderOnly();
