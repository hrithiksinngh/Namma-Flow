"""Tests for the synthetic rainfall generators (``src/data_pipeline/weather_synthetic.py``):
the areal / point climatologies (R1-04, R1-05) and design storms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.weather import design_storm, generate_synthetic_weather
from src.utils.config import ConfigError, deep_merge
from tests.conftest import TZ


# --------------------------------------------------------------------------- synthetic climatology


@pytest.mark.unit
def test_synthetic_weather_is_deterministic_and_schema_valid(cfg):
    a = generate_synthetic_weather("2022-01-01", "2022-03-31", cfg, seed=5)
    b = generate_synthetic_weather("2022-01-01", "2022-03-31", cfg, seed=5)
    c = generate_synthetic_weather("2022-01-01", "2022-03-31", cfg, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not a["precipitation_mm"].equals(c["precipitation_mm"]) or a["precipitation_mm"].sum() == 0
    assert len(a) == 90 * 24 and a.index.freqstr == "h"
    assert set(a["source"]) == {"synthetic"} and not a["is_imputed"].any()
    assert (a["precipitation_mm"] >= 0).all()


@pytest.mark.unit
def test_synthetic_weather_sub_range_consistency(cfg):
    full = generate_synthetic_weather("2022-05-01", "2022-10-31", cfg)
    part = generate_synthetic_weather("2022-07-10 06:00", "2022-08-20", cfg)
    pd.testing.assert_frame_equal(part, full.loc[part.index[0] : part.index[-1]], check_freq=False)


@pytest.mark.integration
def test_synthetic_weather_matches_bengaluru_climatology(cfg):
    point = deep_merge(cfg, {"weather": {"synthetic_scale": "point"}})
    frame = generate_synthetic_weather("2010-01-01", "2019-12-31", point, seed=11)
    rain = frame["precipitation_mm"]
    annual = rain.groupby(rain.index.year).sum()
    assert 850 <= annual.mean() <= 1100, annual.to_dict()
    monthly = rain.groupby(rain.index.month).sum() / 10
    assert monthly[[9, 10]].mean() > 3 * monthly[[1, 2, 3]].mean()
    assert monthly[[9, 10]].mean() > monthly[[6, 7]].mean(), "Sep-Oct is the main peak"
    by_hour = rain.groupby(rain.index.hour).sum()
    assert by_hour.loc[14:23].sum() / by_hour.sum() > 0.55, "convective afternoon/evening maximum"
    daily = rain.resample("D").sum()
    assert (daily >= 60).sum() >= 3 and daily.max() <= 160
    assert rain.max() <= cfg["weather"]["max_precip_mm_h"]
    wet_hours_per_day = (rain > 0).resample("D").sum()
    assert wet_hours_per_day[daily > 0].median() <= 6


@pytest.mark.unit
def test_synthetic_weather_validation(cfg):
    with pytest.raises(ValueError):
        generate_synthetic_weather("2022-02-01", "2022-01-01", cfg)
    with pytest.raises(ValueError):
        generate_synthetic_weather("2022-01-01", "2022-01-02", cfg, seed=-1)
    with pytest.raises(ConfigError, match="synthetic_scale"):
        generate_synthetic_weather("2022-01-01", "2022-01-02", deep_merge(cfg, {"weather": {"synthetic_scale": "gauge"}}))


def _rain_stats(rain: pd.Series) -> dict:
    n_years = rain.index.year.nunique()
    wet = rain[rain > 0.1]
    daily = rain.resample("D").sum()
    six_h = rain.rolling(6).sum()
    return {
        "annual": rain.sum() / n_years,
        "wet_hour_frac": len(wet) / len(rain),
        "q99_wet": wet.quantile(0.99),
        "max": rain.max(),
        "six_h_ge_20_per_yr": (six_h >= 20).sum() / n_years,
        "days_ge_50_per_yr": (daily >= 50).sum() / n_years,
        "wet_days_per_yr": (daily >= 1).sum() / n_years,
    }


@pytest.mark.integration
def test_synthetic_default_scale_matches_reanalysis_record(cfg):
    """R1-04: the offline fallback must look like the ERA5 areal record the pipeline is calibrated on.

    Targets from the real 2018-2024 Open-Meteo record at the corridor: ~1009 mm/yr, 11 % wet
    hours, wet-hour p99 7.0 mm/h, max 20.6 mm/h, 10 six-hour windows >= 20 mm and 0.4 days >= 50 mm
    per year, ~142 wet days per year. The gauge-scale generator gave 2.3 % wet hours, p99 38 mm/h,
    81 six-hour windows >= 20 mm and 2.9 days >= 50 mm per year.
    """
    stats = _rain_stats(generate_synthetic_weather("1995-01-01", "2024-12-31", cfg)["precipitation_mm"])
    assert 850 <= stats["annual"] <= 1150, stats
    assert 0.07 <= stats["wet_hour_frac"] <= 0.16, stats
    assert 4.0 <= stats["q99_wet"] <= 10.0, stats
    assert 8.0 <= stats["max"] <= 35.0, stats
    assert stats["six_h_ge_20_per_yr"] <= 40, stats
    assert stats["days_ge_50_per_yr"] <= 1.5, stats
    assert 100 <= stats["wet_days_per_yr"] <= 180, stats
    point = _rain_stats(generate_synthetic_weather(
        "1995-01-01", "2024-12-31", deep_merge(cfg, {"weather": {"synthetic_scale": "point"}}))["precipitation_mm"])
    assert point["q99_wet"] > 3 * stats["q99_wet"], "the point scale keeps gauge-like intensities"


@pytest.mark.integration
@pytest.mark.parametrize("scale", ["areal", "point"])
def test_synthetic_bursts_spill_into_the_next_month(cfg, scale):
    """R1-05: late bursts on a month's last day continue into the next month (no month-end pile-up)."""
    scaled = deep_merge(cfg, {"weather": {"synthetic_scale": scale}})
    rain = generate_synthetic_weather("1960-01-01", "2059-12-31", scaled)["precipitation_mm"]
    idx = rain.index
    late = idx.hour == 23
    month_end = rain[late & idx.is_month_end].mean()
    other = rain[late & ~idx.is_month_end].mean()
    assert month_end < 2.0 * other, (month_end, other)
    first_night = rain[(idx.day == 1) & (idx.hour < 6)]
    assert first_night.sum() > 0, "the previous month's nocturnal bursts must reach day 1"
    other_nights = rain[(idx.day != 1) & (idx.hour < 6)].mean()
    assert 0.5 * other_nights < first_night.mean() < 2.0 * other_nights
    # still identical to a sub-range generated alone, across the month boundary
    part = generate_synthetic_weather("2022-08-01", "2022-08-02", scaled)
    full = generate_synthetic_weather("2022-07-01", "2022-08-31", scaled)
    pd.testing.assert_frame_equal(part, full.loc[part.index[0] : part.index[-1]], check_freq=False)
    total = generate_synthetic_weather("2000-01-01", "2009-12-31", scaled)["precipitation_mm"].sum()
    by_month = sum(
        generate_synthetic_weather(f"{y}-{m:02d}-01", (pd.Timestamp(f"{y}-{m:02d}-01") + pd.offsets.MonthEnd(0)).date(),
                                   scaled)["precipitation_mm"].sum()
        for y in (2003,) for m in range(1, 13)
    )
    assert by_month == pytest.approx(
        generate_synthetic_weather("2003-01-01", "2003-12-31", scaled)["precipitation_mm"].sum()
    )
    assert total > 0


# --------------------------------------------------------------------------- design storms


@pytest.mark.unit
@pytest.mark.parametrize("shape", ["chicago", "uniform", "triangular"])
def test_design_storm_conserves_total(shape):
    out = design_storm(130.0, 6, start="2026-09-23 18:00", lead_hours=24, horizon_hours=48, shape=shape)
    assert len(out) == 72 and out.index.freqstr == "h" and str(out.index.tz) == TZ
    assert out["precipitation_mm"].sum() == pytest.approx(130.0, rel=1e-12, abs=1e-12)
    storm = out["precipitation_mm"].to_numpy()
    assert not storm[:24].any() and not storm[30:].any() and (storm[24:30] > 0).all()
    assert out.index[24] == pd.Timestamp("2026-09-23 18:00", tz=TZ)
    assert out["is_forecast"].tolist() == [False] * 24 + [True] * 48
    assert set(out["source"]) == {"design_storm"} and not out["is_imputed"].any()


@pytest.mark.unit
def test_design_storm_chicago_peak_position():
    early = design_storm(80.0, 10, start="2026-09-23 00:00", lead_hours=0, horizon_hours=10, peak_position=0.2)
    late = design_storm(80.0, 10, start="2026-09-23 00:00", lead_hours=0, horizon_hours=10, peak_position=0.8)
    assert int(np.argmax(early["precipitation_mm"])) in (1, 2)
    assert int(np.argmax(late["precipitation_mm"])) in (7, 8)
    for r in (0.0, 1.0):
        edge = design_storm(50.0, 5, start="2026-09-23", lead_hours=2, horizon_hours=6, peak_position=r)
        assert edge["precipitation_mm"].sum() == pytest.approx(50.0)
    peak = design_storm(130.0, 6, start="2026-09-23 18:00")["precipitation_mm"].max()
    assert 30.0 < peak < 90.0


@pytest.mark.unit
def test_design_storm_edge_cases_and_validation():
    one = design_storm(30.0, 1, start=pd.Timestamp("2026-09-23 18:30", tz="UTC"), lead_hours=0, horizon_hours=1)
    assert one["precipitation_mm"].tolist() == [30.0]
    assert one.index[0] == pd.Timestamp("2026-09-24 00:00", tz=TZ), "start converted to tz and floored"
    zero = design_storm(0.0, 3, start="2026-09-23")
    assert zero["precipitation_mm"].sum() == 0.0
    bad_calls = [
        dict(total_mm=-1, duration_h=3),
        dict(total_mm=float("nan"), duration_h=3),
        dict(total_mm=10, duration_h=0),
        dict(total_mm=10, duration_h=2.5),
        dict(total_mm=10, duration_h=60),
        dict(total_mm=10, duration_h=3, shape="zigzag"),
        dict(total_mm=10, duration_h=3, peak_position=1.5),
        dict(total_mm=10, duration_h=3, lead_hours=-1),
        dict(total_mm=10, duration_h=3, tz="Mars/Olympus"),
    ]
    for kwargs in bad_calls:
        total, duration = kwargs.pop("total_mm"), kwargs.pop("duration_h")
        with pytest.raises(ValueError):
            design_storm(total, duration, start="2026-09-23", **kwargs)
