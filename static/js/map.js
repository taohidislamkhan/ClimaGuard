/* Leaflet map of Bangladesh divisions (GeoJSON outline, no tile server). */
const RiskMap = (() => {
  let geo = null;
  // Label placement that keeps the eight names from colliding.
  const LABEL_DIR = { Rangpur: "top", Rajshahi: "left", Mymensingh: "top", Sylhet: "right",
                      Dhaka: "right", Khulna: "left", Barisal: "bottom", Chattogram: "right" };

  async function loadGeo() {
    if (!geo) geo = await (await fetch("/static/geo/bd_divisions.geojson")).json();
    return geo;
  }

  async function render(elId, points, { interactive = false, labels = true, radius = 6 } = {}) {
    const el = document.getElementById(elId);
    if (el._map) el._map.remove();
    const map = L.map(el, {
      zoomControl: interactive, attributionControl: false, scrollWheelZoom: interactive,
      dragging: interactive, doubleClickZoom: interactive, boxZoom: false, keyboard: false,
      zoomSnap: 0.1,
    });
    el._map = map;

    const byName = Object.fromEntries(points.map((p) => [p.name, p]));
    const layer = L.geoJSON(await loadGeo(), {
      style: (f) => {
        const p = byName[f.properties.name];
        const c = p ? RISK[p.level].c : "#9DB7D5";
        return { color: cssVar("--card"), weight: 1.2, fillColor: interactive ? c : cssVar("--map-fill"),
                 fillOpacity: interactive ? 0.28 : 0.9 };
      },
    }).addTo(map);
    // Extra horizontal padding keeps the permanent labels inside the frame.
    map.fitBounds(layer.getBounds(), { padding: interactive ? [20, 20] : [34, 6] });

    points.forEach((p) => {
      const m = L.circleMarker([p.lat, p.lon], {
        radius, color: cssVar("--card"), weight: 1.5, fillColor: RISK[p.level].c, fillOpacity: 1,
      }).addTo(map);
      m.bindPopup(`<b>${esc(p.name)}</b><br>${esc(p.level)} · score ${p.score}/100`);
      if (labels) {
        const dir = LABEL_DIR[p.name] || "right";
        const off = { right: [4, 0], left: [-4, 0], top: [0, -4], bottom: [0, 4] }[dir];
        m.bindTooltip(esc(p.name), { permanent: true, direction: dir, offset: off,
                                     className: "div-label" });
      }
    });
    if (interactive) {
      L.control.attribution({ prefix: false })
        .addAttribution("Boundaries © geoBoundaries (CC BY 4.0)").addTo(map);
    }
    return map;
  }

  return { render };
})();
