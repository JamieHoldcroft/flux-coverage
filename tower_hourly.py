"""
tower_hourly.py
===============

Step 1 (laptop). Turns the 42 Australian tower met files into the tower side
of the comparison: every MEASURED value (qc == 0), in friendly units, on
hours that line up with ERA5 hours (UTC).

    python tower_hourly.py --met C:/Users/jamie/CCRC/FLUXNET/Met --out data

What it does, per site:
  1. Reads each variable and keeps only measured steps (qc == 0). Gap-filled
     steps (qc 1-5) are thrown away.
  2. Throws away values outside physical bounds (BOUNDS below) and the known
     faulty sensor years (BAD_YEARS: dead rain gauges, a pressure sensor ~100
     hPa off, etc.). Every removal is counted in
     data/tower_cleaning.csv.
  3. Converts the local standard time stamps (the time_zone attribute of the
     file) to UTC.
  4. Hourly table: averages the two half-hours that make up each UTC hour
     (HH:00 and HH:30 UTC). An hour is kept only if BOTH half-hours are
     measured, so a value is never half measured, half missing. Sites with
     60-minute steps (AU-Otw, AU-Tum) are used as they are.
  5. Half-hourly table: the raw measured steps (UTC), for the sensitivity
     check that asks whether averaging to hours hides tower extremes.

Units (both tables):
    tair   deg C       Tair - 273.15
    vpd    hPa         VPD (already hPa in these files)
    qair   g/kg        Qair * 1000
    psurf  hPa         Psurf / 100
    swdown W/m2        SWdown (night values are measured zeros)
    lwdown W/m2        LWdown
    precip mm/h        Precip (kg/m2/s) * 3600, a RATE, so half-hours and hours compare
    wind   m/s         Wind (at the tower's measurement height, not 10 m)

Outputs (in --out):
    tower_hourly.csv.gz       site, time (UTC start of the hour), the 8 variables
    tower_halfhourly.csv.gz   site, time (UTC start of the step), the 8 variables
    tower_sites.csv           one row per site: location, heights, time zone, period
    tower_cleaning.csv        per site and variable: steps, measured, removed and why
"""

import argparse
import glob
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

# file variable -> (output name, conversion to output units)
VARIABLES = {
    "Tair":   ("tair",   lambda x: x - 273.15),
    "VPD":    ("vpd",    lambda x: x),
    "Qair":   ("qair",   lambda x: x * 1000.0),
    "Psurf":  ("psurf",  lambda x: x / 100.0),
    "SWdown": ("swdown", lambda x: x),
    "LWdown": ("lwdown", lambda x: x),
    "Precip": ("precip", lambda x: x * 3600.0),
    "Wind":   ("wind",   lambda x: x),
}
NAMES = [v[0] for v in VARIABLES.values()]
# Qair in these files is not a separate sensor: FluxnetLSM derives it from
# VPD_F_MDS and Tair. So it counts as measured only when both of those are.
QC_FROM = {"Qair": ["Tair", "VPD"]}

# Physical bounds in output units. Values outside are sensor faults, not weather.
# Small negative SW/VPD/rain (sensor offsets) are set to 0 instead of thrown away.
BOUNDS = {"tair": (-25, 55), "vpd": (0, 120), "qair": (0, 40), "psurf": (600, 1100),
          "swdown": (0, 1500), "lwdown": (100, 650), "precip": (0, 300), "wind": (0, 40)}
ZERO_SMALL_NEGATIVES = {"swdown": -20.0, "vpd": -1.0, "precip": -0.1}

# Site-years whose values are flagged as measured but are clearly a sensor
# fault. Each was found by looking at every site's min / 0.1% / 99.9% / max
# (tower_check printout in the README) or in the earlier CABLE screening:
#   rain:  dead or blocked gauges, annual total a fraction of normal
#   Rgf psurf: about 100 hPa too low for months (known from screening)
#   Boy, Cow lwdown: hundreds of hours above 500 W/m2 in the last years
#   DaS 2019/2021 Tair (known from screening): humidity below 2 g/kg in Darwin
#   Sil 2022-2023 humidity: about 1,300 hours a year below 1 g/kg
#   Wac lwdown: not measured, it is exactly sigma*T^4 (found in Run 1 checks)
# Isolated spikes are left in: the main range uses the 0.1 and 99.9
# percentiles of about 5 million hours, so a few dozen bad hours cannot move it.
ALL_YEARS = "all"
BAD_YEARS = {
    ("AU-Rig", "precip"): [2012], ("AU-Ync", "precip"): [2022],
    ("AU-Whr", "precip"): [2017, 2019], ("AU-Stp", "precip"): [2019],
    ("AU-Lon", "precip"): [2024],
    ("AU-Rgf", "psurf"): [2018, 2020, 2021, 2022, 2023],
    ("AU-Boy", "lwdown"): [2024, 2025], ("AU-Cow", "lwdown"): [2025],
    ("AU-DaS", "tair"): [2019, 2021], ("AU-DaS", "vpd"): [2019, 2021], ("AU-DaS", "qair"): [2019, 2021],
    ("AU-Sil", "vpd"): [2022, 2023], ("AU-Sil", "qair"): [2022, 2023],
    ("AU-Wac", "lwdown"): ALL_YEARS,
}


def read_site(path):
    """One met file -> (site info dict, DataFrame of measured values at the
    file's own step, indexed by UTC start time, plus cleaning counts)."""
    ds = xr.open_dataset(path)
    site = ds.attrs["site_code"]
    tz = ds.time.attrs["time_zone"]
    # standard (winter) offset: the files use local STANDARD time all year
    offset_h = ZoneInfo(tz).utcoffset(pd.Timestamp("2021-07-01").to_pydatetime()).total_seconds() / 3600
    local = pd.DatetimeIndex(ds.time.values)
    step_min = int((local[1] - local[0]) / pd.Timedelta(minutes=1))
    utc = local - pd.Timedelta(hours=offset_h)

    cols, cleaning = {}, []
    for var, (name, convert) in VARIABLES.items():
        x = convert(ds[var].values.reshape(-1).astype("float64"))
        n_total = len(x)
        measured = np.isfinite(x)
        for source in QC_FROM.get(var, [var]):
            measured &= ds[f"{source}_qc"].values.reshape(-1) == 0
        x = np.where(measured, x, np.nan)
        if name in ZERO_SMALL_NEGATIVES:
            small = (x < 0) & (x >= ZERO_SMALL_NEGATIVES[name])
            x[small] = 0.0
        lo, hi = BOUNDS[name]
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        x[bad] = np.nan
        n_bad_years = 0
        bad_years = BAD_YEARS.get((site, name))
        if bad_years is not None:
            in_years = np.ones(len(x), bool) if bad_years == ALL_YEARS else np.isin(local.year, bad_years)
            in_bad = in_years & np.isfinite(x)
            n_bad_years = int(in_bad.sum())
            x[in_bad] = np.nan
        cols[name] = x
        cleaning.append({"site": site, "variable": name, "steps": n_total,
                         "measured": int(measured.sum()), "removed_bounds": int(bad.sum()),
                         "removed_bad_years": n_bad_years, "kept": int(np.isfinite(x).sum())})

    df = pd.DataFrame(cols, index=pd.DatetimeIndex(utc, name="time"))
    info = {"site": site, "name": ds.attrs.get("site_name", ""),
            "lat": float(ds.latitude.values.reshape(-1)[0]),
            "lon": float(ds.longitude.values.reshape(-1)[0]),
            "elevation": float(ds.elevation.values.reshape(-1)[0]),
            "reference_height": float(ds.reference_height.values.reshape(-1)[0]),
            "canopy_height": float(ds.canopy_height.values.reshape(-1)[0]),
            "igbp": str(ds.IGBP_veg_short.values), "time_zone": tz, "utc_offset": offset_h,
            "step_min": step_min, "first_utc": utc[0], "last_utc": utc[-1]}
    ds.close()
    return info, df, cleaning


def to_hourly(df, step_min):
    """Average the measured steps of each UTC hour; keep an hour only when
    every step in it is measured (2 half-hours, or the single 60-min step)."""
    if step_min == 60:
        if (df.index.minute != 0).any():
            raise ValueError("60-minute file whose steps do not start on the UTC hour")
        return df.copy()
    if step_min != 30:
        raise ValueError(f"unexpected step {step_min} min")
    hour = df.index.floor("h")
    g = df.groupby(hour)
    mean, count = g.mean(), g.count()
    return mean.where(count == 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--met", required=True, help="folder with the AU-*_Met.nc files")
    p.add_argument("--out", default="data")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.met) / "AU-*_Met.nc")))
    if not files:
        raise SystemExit(f"no AU-*_Met.nc files in {args.met}")
    infos, hourly, halfhourly, cleaning = [], [], [], []
    for f in files:
        info, df, clean = read_site(f)
        hr = to_hourly(df, info["step_min"])
        for frame, store in ((df, halfhourly), (hr, hourly)):
            frame = frame.dropna(how="all").reset_index()
            frame.insert(0, "site", info["site"])
            store.append(frame)
        info["hours_any_measured"] = int(hr.notna().any(axis=1).sum())
        infos.append(info)
        cleaning.extend(clean)
        print(f"{info['site']}: {len(df):,} steps, {info['hours_any_measured']:,} hours with a measured value", flush=True)

    sites = pd.DataFrame(infos)
    sites.to_csv(out / "tower_sites.csv", index=False)
    pd.DataFrame(cleaning).to_csv(out / "tower_cleaning.csv", index=False)
    for frames, name in ((hourly, "tower_hourly.csv.gz"), (halfhourly, "tower_halfhourly.csv.gz")):
        table = pd.concat(frames, ignore_index=True)
        table["time"] = table["time"].dt.strftime("%Y-%m-%d %H:%M")
        table.to_csv(out / name, index=False, float_format="%.6g")
        print(f"{name}: {len(table):,} rows; measured values per variable:")
        print("   " + "  ".join(f"{c} {table[c].notna().sum():,}" for c in NAMES))


if __name__ == "__main__":
    main()
