"""Environment page — climate & air-quality trends for the selected country."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header, env_cell, html_grid
from components.theme import MUTED, CYAN_DARK
from utils.predict import _load_clean


def render(state: dict) -> None:
    country = state["country"]
    date = state["date"]
    page_header(country["country_name"], country["region"], date)

    # We pull the full history for this country from cleaned_data.csv.
    full = _load_clean()
    full["date"] = pd.to_datetime(full["date"])
    sub = full[full["country_code"] == country["country_code"]].sort_values("date")

    if sub.empty:
        st.warning("No historical environmental data for this country.")
        return

    # Snap to the selected week, and only show history up to it.
    pos = int((sub["date"] - date).abs().to_numpy().argmin())
    sub = sub.iloc[:pos + 1]
    cur = sub.iloc[-1]

    cols = [
        env_cell("🌡️", "Temperature",     f"{cur['temperature_celsius']:.1f} °C",
                 f"max {sub['temperature_celsius'].max():.1f}, min {sub['temperature_celsius'].min():.1f}"),
        env_cell("🌡️", "Temp anomaly",    f"{cur['temp_anomaly_celsius']:+.2f} °C",
                 "vs climatology"),
        env_cell("🌫️", "PM2.5",            f"{cur['pm25_ugm3']:.1f} µg/m³",
                 f"avg {sub['pm25_ugm3'].mean():.1f}"),
        env_cell("🌀", "AQI",              f"{cur['air_quality_index']:.0f}",
                 f"max {sub['air_quality_index'].max():.0f}"),
        env_cell("🌧️", "Precipitation",   f"{cur['precipitation_mm']:.1f} mm",
                 "weekly"),
        env_cell("☀️", "Heat-wave days",  f"{int(cur['heat_wave_days'])}",
                 f"max {int(sub['heat_wave_days'].max())}"),
        env_cell("⚡", "Extreme weather", f"{int(cur['extreme_weather_events'])}",
                 f"max {int(sub['extreme_weather_events'].max())}"),
        env_cell("🏥", "Healthcare",       f"{cur['healthcare_access_index']:.0f}",
                 "0-100 scale"),
    ]
    st.markdown(html_grid(cols, columns=4, gap=12, extra_class="cg-env-grid"),
                unsafe_allow_html=True)
    st.markdown(f'<div class="cg-footnote">Week of {cur["date"].strftime("%b %d, %Y")}; '
                'max / min / avg are over the country\'s history up to that week.</div>',
                unsafe_allow_html=True)

    st.write("")

    # Time-series: Temperature & anomaly
    fig = px.line(sub.tail(52), x="date", y=["temperature_celsius", "temp_anomaly_celsius"],
                  labels={"value": "°C", "date": "", "variable": "Series"},
                  color_discrete_sequence=[CYAN_DARK, "#F59E0B"])
    fig.update_layout(height=280, margin=dict(t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                      legend=dict(orientation="h", y=1.06, x=0))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="cg-footnote">52 weeks of temperature '
                'and anomaly up to the selected week.</div>',
                unsafe_allow_html=True)

    # PM2.5 + AQI
    sub_poll = sub.tail(52)
    fig = px.line(sub_poll, x="date", y="pm25_ugm3",
                  color_discrete_sequence=[CYAN_DARK],
                  labels={"pm25_ugm3": "PM2.5 (µg/m³)", "date": ""})
    fig.update_layout(height=260, margin=dict(t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=11, color=MUTED))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Precipitation + heat waves
    fig = px.bar(sub.tail(52), x="date", y="precipitation_mm",
                 color_discrete_sequence=[CYAN_DARK],
                 labels={"precipitation_mm": "mm", "date": ""})
    fig.update_layout(height=240, margin=dict(t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=11, color=MUTED))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
