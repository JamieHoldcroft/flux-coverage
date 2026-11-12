"""
era5_hourly_australia.py
========================

Step 2 (Gadi). Pulls hourly ERA5 (NCI project rt52) for every Australian land
cell and stores it, one file per year, in the same units as the tower table.

    # once, on the login node (seconds): which cells are Australian land
    python3 era5_hourly_australia.py --make-cells --out /scratch/dt6/jh6400/fc/era5_hourly
    # then the years (normally through run_era5_hourly.pbs)
    python3 era5_hourly_australia.py --years 1980 2025 --workers 12 --out /scratch/dt6/jh6400/fc/era5_hourly

Cells (cells.csv): ERA5 0.25 deg cells in the box 112-154 E, 10-44 S with
land-sea mask >= 0.5, minus the cells that belong to Indonesia, Timor and
Papua New Guinea (north of 11.2 S west of 127.5 E, and north of 11.8 S east
of 146 E; mainland Australia, Tasmania and the Torres Strait islands are
all outside those two corners). Each cell gets an area weight cos(lat) and
its local standard time offset (WA +8, NT and SA +9.5, the rest +10).

Hours: each output hour is the hour STARTING at its time stamp (UTC), so it
lines up with the tower hours from tower_hourly.py.
  - Rain, sunshine and longwave (tp, ssrd, strd) are totals for the hour
    ENDING at their ERA5 stamp, so hour [t, t+1) takes the value stamped t+1.
  - Temperature, dewpoint, pressure and wind are snapshots AT the stamp, so
    hour [t, t+1) takes the average of the snapshots at t and t+1 (the best
    estimate of the hour mean, which is what a tower records).
  - The last hour of the last year needs the 00 UTC stamp of the next 1 January;
    if that file does not exist the hour is left empty.

Variables (same names and units as tower_hourly.py):
    tair   deg C    2t - 273.15
    vpd    hPa      es(2t) - es(2d), Tetens, es = 6.108 exp(17.27 T / (T + 237.3))
    qair   g/kg     1000 * 0.622 e / (sp - 0.378 e), e = es(2d)
    psurf  hPa      sp / 100
    swdown W/m2     ssrd / 3600
    lwdown W/m2     strd / 3600
    precip mm/h     tp * 1000
    wind   m/s      sqrt(10u^2 + 10v^2)
VPD, humidity and wind are worked out at each snapshot, then averaged.

Output: <out>/era5_hourly_<year>.nc, dims (time, cell), packed to 16-bit
integers (rain kept as 32-bit floats because it spans six orders of size),
plus <out>/cells.csv.
"""

import argparse
import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RT52 = "/g/data/rt52/era5/single-levels/reanalysis"
BOX = {"lat": (-10.0, -44.0), "lon": (112.0, 154.0)}      # ERA5 latitude runs north -> south
INPUTS = ["2t", "2d", "sp", "10u", "10v", "tp", "ssrd", "strd"]
ACCUMULATED = {"tp", "ssrd", "strd"}
OUT_VARS = ["tair", "vpd", "qair", "psurf", "swdown", "lwdown", "precip", "wind"]
# 16-bit packing: value = stored * scale + offset. Steps are far below sensor
# precision (0.01 deg C, 0.005 hPa VPD, ...) and the ranges cover every
# value Australia can have.
PACKING = {"tair": (0.01, 10.0), "vpd": (0.005, 160.0), "qair": (0.001, 30.0),
           "psurf": (0.02, 800.0), "swdown": (0.05, 1600.0), "lwdown": (0.02, 400.0),
           "wind": (0.002, 64.0)}                         # precip: float32, no packing


def month_files(var, year):
    files = sorted(glob.glob(f"{RT52}/{var}/{year}/{var}_era5_oper_sfc_*.nc"))
    if not files:
        raise FileNotFoundError(f"no ERA5 files for {var} {year} under {RT52}/{var}/{year}")
    return files


def utc_offset(lat, lon):
    """Local standard time offset (hours) from the state borders, by cell centre.
    WA west of 129 E; NT and SA 129-138 E, and SA 138-141 E south of 26 S;
    Queensland 138-141 E north of 26 S; everything east of 141 E +10."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    off = np.full(lat.shape, 10.0)
    off[lon < 129] = 8.0
    off[(lon >= 129) & (lon < 138)] = 9.5
    off[(lon >= 138) & (lon < 141) & (lat <= -26)] = 9.5
    return off


def make_cells(out_dir):
    """Australian land cells -> cells.csv (cell, i_lat, i_lon, lat, lon, weight, utc_offset)."""
    files = sorted(glob.glob(f"{RT52}/lsm/*/lsm_era5_oper_sfc_*.nc"))
    if not files:
        raise FileNotFoundError(f"no land-sea mask under {RT52}/lsm")
    with xr.open_dataset(files[0]) as ds:
        lsm = ds["lsm"].isel(time=0).sel(latitude=slice(*BOX["lat"]), longitude=slice(*BOX["lon"])).load()
    lat2, lon2 = np.meshgrid(lsm.latitude.values, lsm.longitude.values, indexing="ij")
    land = lsm.values >= 0.5
    foreign = ((lat2 > -11.2) & (lon2 < 127.5)) | ((lat2 > -11.8) & (lon2 >= 146.0))
    keep = land & ~foreign
    i_lat, i_lon = np.nonzero(keep)
    cells = pd.DataFrame({"cell": np.arange(len(i_lat)), "i_lat": i_lat, "i_lon": i_lon,
                          "lat": lat2[keep], "lon": lon2[keep]})
    cells["weight"] = np.cos(np.radians(cells.lat))
    cells["utc_offset"] = utc_offset(cells.lat, cells.lon)
    cells.to_csv(Path(out_dir) / "cells.csv", index=False)
    print(f"{len(cells)} Australian land cells ({int(land.sum() - keep.sum())} foreign land cells dropped) "
          f"-> {Path(out_dir) / 'cells.csv'}")
    return cells


def read_stamps(var, year, cells, next_first=True):
    """All hourly stamps of one variable for one year at the cells, plus the
    first stamp of the next year (needed to close the last hour).
    Returns (times, values (stamps, cells) float32)."""
    times, vals = [], []
    for path in month_files(var, year):
        with xr.open_dataset(path) as ds:
            da = ds[list(ds.data_vars)[0]].sel(latitude=slice(*BOX["lat"]), longitude=slice(*BOX["lon"]))
            a = da.values                                  # (hours, lat, lon), decoded
            times.append(da.time.values)
            vals.append(a[:, cells.i_lat.values, cells.i_lon.values].astype("float32"))
    if next_first:
        try:
            path = month_files(var, year + 1)[0]
            with xr.open_dataset(path) as ds:
                da = ds[list(ds.data_vars)[0]].isel(time=slice(0, 1))
                da = da.sel(latitude=slice(*BOX["lat"]), longitude=slice(*BOX["lon"]))
                a = da.values
                times.append(da.time.values)
                vals.append(a[:, cells.i_lat.values, cells.i_lon.values].astype("float32"))
        except FileNotFoundError:
            pass
    return np.concatenate(times), np.concatenate(vals)


def es_hpa(t_c):
    """Saturation vapour pressure (hPa), Tetens."""
    return 6.108 * np.exp(17.27 * t_c / (t_c + 237.3))


def build_year(year, cells, out_dir):
    target = Path(out_dir) / f"era5_hourly_{year}.nc"
    if target.exists():
        return f"{year}: already done"
    hours = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    stamps = hours.append(pd.DatetimeIndex([hours[-1] + pd.Timedelta(hours=1)]))   # n_hours + 1 stamps
    raw = {}
    for var in INPUTS:
        t, v = read_stamps(var, year, cells)
        pos = pd.DatetimeIndex(t).get_indexer(stamps)      # -1 where a stamp is missing
        if (pos[:-1] < 0).any():
            missing = stamps[:-1][pos[:-1] < 0]
            raise ValueError(f"{var} {year}: {len(missing)} hourly stamps missing, first {missing[0]}")
        full = np.full((len(stamps), len(cells)), np.nan, np.float32)
        ok = pos >= 0
        full[ok] = v[pos[ok]]
        raw[var] = full

    def hour_mean(a):                                      # snapshots t and t+1 -> hour [t, t+1)
        return 0.5 * (a[:-1] + a[1:])

    def hour_total(a):                                     # total stamped t+1 -> hour [t, t+1)
        return a[1:]

    t_c, td_c = raw["2t"] - 273.15, raw["2d"] - 273.15
    e = es_hpa(td_c)                                       # actual vapour pressure, hPa
    p_hpa = raw["sp"] / 100.0
    out = {
        "tair": hour_mean(t_c),
        "vpd": hour_mean(np.clip(es_hpa(t_c) - e, 0, None)),
        "qair": hour_mean(1000.0 * 0.622 * e / (p_hpa - 0.378 * e)),
        "psurf": hour_mean(p_hpa),
        "swdown": np.clip(hour_total(raw["ssrd"]) / 3600.0, 0, None),
        "lwdown": hour_total(raw["strd"]) / 3600.0,
        "precip": np.clip(hour_total(raw["tp"]) * 1000.0, 0, None),
        "wind": hour_mean(np.sqrt(raw["10u"] ** 2 + raw["10v"] ** 2)),
    }
    del raw
    ds = xr.Dataset({k: (("time", "cell"), v.astype("float32")) for k, v in out.items()},
                    coords={"time": hours.values, "cell": cells.cell.values})
    for k, unit in zip(OUT_VARS, ["degC", "hPa", "g/kg", "hPa", "W/m2", "W/m2", "mm/h", "m/s"]):
        ds[k].attrs["units"] = unit
    ds.attrs["hours"] = "each time is the START of a one-hour period, UTC"
    enc = {}
    for k in OUT_VARS:
        chunks = (min(744, len(hours)), len(cells))
        if k in PACKING:
            scale, offset = PACKING[k]
            enc[k] = {"dtype": "int16", "scale_factor": scale, "add_offset": offset,
                      "_FillValue": np.int16(-32768), "zlib": True, "complevel": 1, "chunksizes": chunks}
        else:
            enc[k] = {"dtype": "float32", "zlib": True, "complevel": 1, "chunksizes": chunks}
    tmp = target.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.rename(target)                                     # only complete files get the real name
    n_empty = int(np.isnan(out["precip"]).all(axis=1).sum())
    return f"{year}: {len(hours)} hours x {len(cells)} cells -> {target} ({n_empty} empty hours)"


def _job(args):
    year, out_dir = args
    cells = pd.read_csv(Path(out_dir) / "cells.csv")
    return build_year(year, cells, out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--years", nargs=2, type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--make-cells", action="store_true")
    p.add_argument("--rt52", default=RT52, help="ERA5 folder (only change for testing)")
    args = p.parse_args()
    globals()["RT52"] = args.rt52
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.make_cells:
        make_cells(out_dir)
    if args.years:
        if not (out_dir / "cells.csv").exists():
            raise SystemExit("run --make-cells first")
        years = list(range(args.years[0], args.years[1] + 1))
        jobs = [(y, str(out_dir)) for y in years]
        if args.workers > 1:
            with Pool(args.workers, initializer=_set_rt52, initargs=(args.rt52,)) as pool:
                for msg in pool.imap_unordered(_job, jobs):
                    print(msg, flush=True)
        else:
            for j in jobs:
                print(_job(j), flush=True)


def _set_rt52(path):
    globals()["RT52"] = path


if __name__ == "__main__":
    main()
