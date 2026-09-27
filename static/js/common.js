/* Shared helpers: location (?loc=), units, API, header, page states. */
const RISK = {
  Low:      { c: "#22A06B", bg: "#E3F6EC" },
  Moderate: { c: "#F5A524", bg: "#FFF3D6" },
  High:     { c: "#E5484D", bg: "#FDE7E8" },
};

const SETTINGS = (window.APP && window.APP.settings) || { units: "C", theme: "light", default_location: "Dhaka" };

/** Read a CSS custom property (theme-aware colours for canvas charts). */
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- location (shared via ?loc=) ---------------- */
const Loc = {
  get() {
    const q = new URLSearchParams(location.search).get("loc");
    return (window.APP?.divisions || []).includes(q) ? q : SETTINGS.default_location;
  },
  /** href with the current ?loc= added (keeps the selection across pages). */
  link(href) {
    const u = new URL(href, location.origin);
    u.searchParams.set("loc", Loc.get());
    return u.pathname + u.search + u.hash;
  },
  set(name) {
    const u = new URL(location.href);
    u.searchParams.set("loc", name);
    location.href = u.toString();
  },
};

/* ---------------- formatting ---------------- */
const fmt = {
  time(iso) {
    return new Date(iso).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "Asia/Dhaka" });
  },
  date(iso) {
    return new Date(iso).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "Asia/Dhaka" });
  },
  shortDate(iso) {
    return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "Asia/Dhaka" });
  },
  num(v, d = 0) {
    return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d);
  },
  /** Temperature in the user's unit (°C stored everywhere). */
  temp(c, d = 0) {
    if (c === null || c === undefined) return "–";
    return SETTINGS.units === "F" ? (c * 9 / 5 + 32).toFixed(d) : Number(c).toFixed(d);
  },
  tempUnit() { return SETTINGS.units === "F" ? "°F" : "°C"; },
};

function greetingFor(date = new Date()) {
  const h = date.getHours();
  if (h < 5) return "Good evening";
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  return "Good evening";
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const msg = body?.detail || body?.error || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return body;
}

/** api() with ?loc= appended. */
function apiLoc(path, opts) {
  const sep = path.includes("?") ? "&" : "?";
  return api(`${path}${sep}loc=${encodeURIComponent(Loc.get())}`, opts);
}

function chip(text, tip, cls = "") {
  return `<span class="chip ${cls}" data-tip="${esc(tip)}">${esc(text)} <i data-lucide="info"></i></span>`;
}

function badge(level, text) {
  return level ? `<span class="badge lvl-${level}">${esc(text ?? level)}</span>` : `<span class="badge">–</span>`;
}

/* ---------------- header ---------------- */
function fillHeader(d) {
  if (d.updated_at) {
    document.getElementById("hdr-date").textContent = fmt.date(d.updated_at);
    document.getElementById("side-updated").textContent = `Today, ${fmt.time(d.updated_at)}`;
  }
  if (d.live === false) {
    document.getElementById("sys-status").textContent = "Offline data";
    document.getElementById("sys-dot").style.background = "#F5A524";
  }
  if (d.profile) setProfileHeader(d.profile);
  if (d.diseases && SETTINGS.notifications?.any_high) {
    document.getElementById("bell-dot").classList.toggle("hidden", !d.diseases.some((x) => x.level === "High"));
  }
}

function setProfileHeader(p) {
  const name = p.name || "Guest";
  document.getElementById("avatar-name").textContent = name;
  document.getElementById("avatar-initial").textContent = name[0].toUpperCase();
}

/* ---------------- page states: skeleton / error / fallback banner ---------------- */
const Page = {
  loading(on) {
    document.body.classList.toggle("is-loading", on);
  },
  error(err, retry) {
    const box = document.getElementById("page-error");
    box.innerHTML = `
      <div class="card error-card flex items-center gap-4 px-5 py-4 mb-3">
        <i data-lucide="triangle-alert" class="w-6 h-6 shrink-0"></i>
        <div class="flex-1 min-w-0">
          <div class="font-semibold text-[var(--heading)] text-[14px]">Couldn't load this page's data</div>
          <div class="text-[12px] mt-0.5 break-words">${esc(err.message || err)}</div>
        </div>
        <button class="btn-primary" id="retry-btn"><i data-lucide="refresh-cw"></i> Retry</button>
      </div>`;
    box.classList.remove("hidden");
    document.getElementById("retry-btn").onclick = () => { box.classList.add("hidden"); retry(); };
    lucide.createIcons();
  },
  banner(data) {
    const b = document.getElementById("fallback-banner");
    const fallback = data && data.live === false;
    b.classList.toggle("hidden", !fallback);
    if (fallback) document.getElementById("fallback-date").textContent = data.data_date || data.updated_at || "";
  },
  /**
   * Run a page loader with skeletons, error card + Retry, and the fallback banner.
   * loader: async () => data ; render: (data) => void
   */
  async run(loader, render) {
    const go = async () => {
      Page.loading(true);
      try {
        const data = await loader();
        Page.banner(data);
        render(data);
        document.body.dataset.ready = "1";
      } catch (e) {
        console.error(e);
        Page.error(e, go);
      } finally {
        Page.loading(false);
        lucide.createIcons();
      }
    };
    return go();
  },
};

/* ---------------- boot: location selector + links keep ?loc= ---------------- */
(function boot() {
  const sel = document.getElementById("loc-select");
  if (sel) {
    sel.value = Loc.get();
    sel.addEventListener("change", () => Loc.set(sel.value));
  }
  document.querySelectorAll("a[data-keep-loc]").forEach((a) => { a.href = Loc.link(a.getAttribute("href")); });
  document.getElementById("hdr-date").textContent = fmt.date(new Date().toISOString());
  api("/api/profile").then(setProfileHeader).catch(() => {});
})();

/* Pages that only need the header use this. */
async function initHeaderOnly() {
  try { fillHeader(await apiLoc("/api/dashboard")); } catch (e) { console.error(e); }
}
