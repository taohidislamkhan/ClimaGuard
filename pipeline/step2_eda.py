"""Step 2 — Exploratory Data Analysis (Phase 3).

Produces, under `output/charts/`:

  1. `hist_numeric.png`         — distribution histograms for every numeric column.
  2. `correlation_heatmap.png`  — Pearson correlation matrix of key climate/health
                                   features.
  3. `scatter_<outcome>_vs_temp.png` — disease outcomes vs. temperature,
                                         coloured by income_level.
  4. `seasonal_<outcome>.png`    — monthly/seasonal line + bar charts.
  5. `yearly_trend_by_region.png` — yearly trend of each climate/heat signal by
                                     region.

Each plot is checked against one of the four hypotheses (see `eda_summary.md`).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/cleaned_data.csv")
OUT_DIR = Path("output/charts")

NUMERIC_HIST_COLS = [
    "temperature_celsius", "temp_anomaly_celsius", "precipitation_mm",
    "heat_wave_days", "drought_indicator", "flood_indicator",
    "extreme_weather_events", "pm25_ugm3", "air_quality_index",
    "respiratory_disease_rate", "cardio_mortality_rate",
    "vector_disease_risk_score", "waterborne_disease_incidents",
    "heat_related_admissions", "healthcare_access_index",
    "gdp_per_capita_usd", "mental_health_index", "food_security_index",
]

CORR_COLS = [
    "temperature_celsius", "pm25_ugm3", "air_quality_index",
    "respiratory_disease_rate", "heat_wave_days", "heat_related_admissions",
    "vector_disease_risk_score", "waterborne_disease_incidents",
    "flood_indicator", "cardio_mortality_rate", "healthcare_access_index",
]

SCATTER_OUTCOMES = [
    "heat_related_admissions",
    "respiratory_disease_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "cardio_mortality_rate",
]

SEASONAL_OUTCOMES = [
    "heat_related_admissions",
    "respiratory_disease_rate",
    "vector_disease_risk_score",
    "waterborne_disease_incidents",
    "cardio_mortality_rate",
]

INCOME_ORDER = ["Low", "Lower-Middle", "Upper-Middle", "High"]
INCOME_COLORS = {
    "Low":          "#d73027",
    "Lower-Middle": "#fc8d59",
    "Upper-Middle": "#91bfdb",
    "High":         "#4575b4",
}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=120)
    plt.close(fig)


def histogram_grid(df: pd.DataFrame) -> None:
    cols = [c for c in NUMERIC_HIST_COLS if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=30, color="#4575b4", edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distribution of numeric features", fontsize=14)
    _save(fig, "hist_numeric.png")


def correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[CORR_COLS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=70, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Correlation heatmap — climate vs. health", fontsize=14)
    _save(fig, "correlation_heatmap.png")
    return corr


def scatter_outcomes_vs_temp(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["temp_bin"] = pd.cut(
        tmp["temperature_celsius"], bins=np.arange(-25, 45, 2), right=False
    )
    for outcome in SCATTER_OUTCOMES:
        agg = (tmp.groupby(["temp_bin", "income_level"], observed=True)[outcome]
                  .mean().reset_index())
        fig, ax = plt.subplots(figsize=(10, 6))
        for inc in INCOME_ORDER:
            sub = agg[agg["income_level"] == inc].dropna(subset=["temp_bin"])
            if sub.empty:
                continue
            centers = [(iv.left + iv.right) / 2 for iv in sub["temp_bin"]]
            ax.plot(centers, sub[outcome], "o-", color=INCOME_COLORS[inc],
                    label=inc, linewidth=2, markersize=5)
        ax.set_title(f"{outcome} vs. temperature (binned mean, by income)", fontsize=13)
        ax.set_xlabel("Temperature (°C, bin midpoint)")
        ax.set_ylabel(f"Mean {outcome}")
        ax.legend(title="Income level")
        ax.grid(alpha=0.3)
        _save(fig, f"scatter_{outcome}_vs_temp.png")


def seasonal_patterns(df: pd.DataFrame) -> None:
    season_order = ["winter", "spring", "summer", "autumn"]
    month_order = list(range(1, 13))
    for outcome in SEASONAL_OUTCOMES:
        monthly = df.groupby("month")[outcome].mean().reindex(month_order)
        seasonal = df.groupby("season")[outcome].mean().reindex(season_order)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(month_order, monthly.values, "o-", color="#4575b4")
        axes[0].set_title(f"{outcome}: monthly mean")
        axes[0].set_xlabel("Month")
        axes[0].set_xticks(month_order)
        axes[0].grid(alpha=0.3)
        axes[1].bar(season_order, seasonal.values,
                    color=["#91bfdb", "#fc8d59", "#d73027", "#4575b4"])
        axes[1].set_title(f"{outcome}: seasonal mean")
        axes[1].set_xlabel("Season (hemisphere-aware)")
        axes[1].grid(alpha=0.3, axis="y")
        _save(fig, f"seasonal_{outcome}.png")


def yearly_trend_by_region(df: pd.DataFrame) -> None:
    metrics = ["temperature_celsius", "heat_related_admissions",
               "pm25_ugm3", "cardio_mortality_rate"]
    regions = sorted(df["region"].dropna().unique().tolist())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)),
                             sharex=True)
    for ax, metric in zip(axes, metrics):
        for region in regions:
            sub = (df[df["region"] == region]
                     .groupby("year")[metric].mean().sort_index())
            ax.plot(sub.index, sub.values, label=region, linewidth=1.6)
        ax.set_title(f"Yearly mean {metric} by region")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Year")
    axes[0].legend(fontsize=8, ncol=4, loc="best")
    _save(fig, "yearly_trend_by_region.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH, parse_dates=["date"])

    print("Step 2 — EDA starting...")
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")

    histogram_grid(df)
    corr = correlation_heatmap(df)
    scatter_outcomes_vs_temp(df)
    seasonal_patterns(df)
    yearly_trend_by_region(df)

    corr.round(3).to_csv(OUT_DIR / "correlation_matrix.csv")

    print("\nHypothesis diagnostics:")
    print("  H1 (temp -> heat admissions)        r =",
          round(corr.loc["temperature_celsius", "heat_related_admissions"], 3))
    print("  H2 (PM2.5 -> respiratory)           r =",
          round(corr.loc["pm25_ugm3", "respiratory_disease_rate"], 3))
    print("  H3 (flood -> waterborne)            r =",
          round(corr.loc["flood_indicator", "waterborne_disease_incidents"], 3))
    print("  H4 (healthcare access -> respir.)   r =",
          round(corr.loc["healthcare_access_index", "respiratory_disease_rate"], 3))

    print(f"\nEDA charts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
