/* Chart.js wrappers: gauge, sparkline, 7-day trend. */
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.color = cssVar("--muted") || "#8A94A6";

const Charts = (() => {
  const registry = {};

  function replace(id, cfg) {
    if (registry[id]) registry[id].destroy();
    registry[id] = new Chart(document.getElementById(id), cfg);
    return registry[id];
  }

  function gauge(id, score, level) {
    const c = RISK[level]?.c || "#8A94A6";
    const canvas = document.getElementById(id);
    const g = canvas.getContext("2d").createLinearGradient(0, 0, 0, 140);
    g.addColorStop(0, c);
    g.addColorStop(1, level === "Moderate" ? "#F9C74F" : c);
    return replace(id, {
      type: "doughnut",
      data: { datasets: [{ data: [score, 100 - score], backgroundColor: [g, "#FDF3DC"],
                           borderWidth: 0, borderRadius: [8, 0] }] },
      options: {
        cutout: "80%", rotation: 0, animation: { duration: 700 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        responsive: false,
      },
    });
  }

  function trackColor(level) {
    return { Low: "#E3F6EC", Moderate: "#FDF3DC", High: "#FDE7E8" }[level] || "#EEF2F7";
  }

  function sparkline(id, values, level) {
    const c = RISK[level]?.c || "#8A94A6";
    const canvas = document.getElementById(id);
    const g = canvas.getContext("2d").createLinearGradient(0, 0, 0, canvas.height || 40);
    g.addColorStop(0, c + "40");
    g.addColorStop(1, c + "00");
    return replace(id, {
      type: "line",
      data: { labels: values.map((_, i) => i),
              datasets: [{ data: values, borderColor: c, borderWidth: 1.8, fill: true,
                           backgroundColor: g, pointRadius: 0, tension: 0.35 }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false, suggestedMin: Math.min(...values) - 8 } },
      },
    });
  }

  /* Draws each point's value just above it. */
  const valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.font = "500 10px Inter, sans-serif";
      ctx.fillStyle = cssVar("--heading");
      ctx.textAlign = "center";
      meta.data.forEach((pt, i) => ctx.fillText(chart.data.datasets[0].data[i], pt.x, pt.y - 8));
      ctx.restore();
    },
  };

  function trend(id, labels, values, tooltipTitles) {
    const canvas = document.getElementById(id);
    const g = canvas.getContext("2d").createLinearGradient(0, 0, 0, 150);
    g.addColorStop(0, "rgba(31,111,235,.14)");
    g.addColorStop(1, "rgba(31,111,235,0)");
    // Scores live in 0–100, so the axis never goes below 0.
    const lo = Math.max(0, Math.min(30, Math.floor((Math.min(...values) - 5) / 10) * 10));
    const hi = Math.max(90, Math.ceil(Math.max(...values) / 10) * 10);
    return replace(id, {
      type: "line",
      data: { labels, datasets: [{ data: values, borderColor: "#1F6FEB", borderWidth: 2,
                                   backgroundColor: g, fill: true, tension: 0.15,
                                   pointRadius: 3.2, pointBackgroundColor: "#1F6FEB",
                                   pointBorderColor: "#fff", pointBorderWidth: 1 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        layout: { padding: { top: 16, right: 6 } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: (items) => tooltipTitles[items[0].dataIndex] } },
        },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { font: { size: 10 } } },
          y: { min: lo, max: hi, ticks: { stepSize: 20, font: { size: 10 } }, afterBuildTicks: (ax) => { ax.ticks = ax.ticks.filter((t) => (t.value - ax.min) % 20 === 0); },
               grid: { color: cssVar("--grid") }, border: { display: false } },
        },
      },
      plugins: [valueLabels],
    });
  }

  return { gauge, sparkline, trend, trackColor };
})();
