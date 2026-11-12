"""
coverage_stats.py
=================

Step 3 (Gadi). Compares, one variable at a time, the weather the 42 towers
measured with all of Australia's ERA5 weather, 1980-2025.

    python3 coverage_stats.py --towers data --era5 /scratch/dt6/jh6400/fc/era5_hourly \
        --years 1980 2025 --workers 8 --out /scratch/dt6/jh6400/fc/results

The four groups (each one step further from the towers):
    A  tower       every fully measured tower hour, all 42 sites pooled
    B  matched     ERA5 at each tower's own cell, at exactly the hours in A
    C  tower cells ERA5 at the towers' cells, every hour 1980-2025
    D  Australia   ERA5 at every Australian land cell, every hour 1980-2025,
                   weighted by cell area (cos latitude)
A vs B = point vs grid (same place, same hours); B vs C = which years and
hours were measured; C vs D = which places have towers.

The tower range of a variable = the 0.1 to 99.9 percentiles of A (the main
range). The headline number is the share of D outside that range ("weather
no tower has measured"). Other ranges are kept for sensitivity: A p1-p99,
A min-max, A from raw half-hours, B and C (ERA5's own view of the tower
places), and A without each site in turn (that site's value to the network).

Variables: tair, vpd, qair, psurf, swdown, lwdown, precip, wind (all hours).
For rain the share of wet hours (>= 0.2 mm/h, about one tip of a tower
gauge) is also reported.

Exactness: shares of D are counted from histograms whose bin edges include
every range end, so they are exact counts, not approximations. Quantiles of
D come from the same histograms (fine bins, e.g. 0.05 deg C), interpolated
within a bin. Everything for A, B and C is computed on the values directly.

Outputs (in --out):
    ranges.csv           per group and quantity: n, min, p0.1 ... p99.9, max, mean, sd, iqr (+ wet share)
    thresholds.csv       every range used: name, quantity, lo, hi
    shares.csv           share below / above / outside each range, for groups A-D
    shares_by_year.csv   D, by local year, main range and ERA5-own (B) range
    shares_by_month.csv  D, by local month (all years)
    shares_by_hour.csv   D, by local hour of day (all years)
    trend.csv            linear trend of the yearly share (percentage points per decade)
    tower_value.csv      share of D outside the range when each site is left out
    point_vs_grid.csv    per site and variable (matched hours): n, means, bias, r, rmse, sds, own p0.1/p99.9
    sites_cells.csv      each tower's ERA5 cell and the distance to it
    per_cell.nc          per cell and year: hours, hours below, hours above the main range
    histograms.nc        per quantity: bin edges and histograms of A, A half-hourly, B, C, D
                         (D also by year x month and by local hour)
    era5_tower_cells.nc  cache of ERA5 at the tower cells (pass 1); reused if present
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = ["tair", "vpd", "qair", "psurf", "swdown", "lwdown", "precip", "wind"]
QUANTITIES = VARIABLES
WET_MM_H = 0.2
PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
MAIN = "tower_p0.1-99.9"
ERA5_OWN = "matched_p0.1-99.9"


def _lin(lo, hi, step):
    return np.round(np.arange(lo, hi + step / 2, step), 10)


_LOG = np.concatenate([[0.0], 10.0 ** (np.arange(-500, 301) / 100.0)])   # 0, then 1e-5 .. 1000, 100 per decade
FINE_EDGES = {"tair": _lin(-40, 60, 0.05), "vpd": _lin(0, 150, 0.05), "qair": _lin(0, 40, 0.01),
              "psurf": _lin(500, 1100, 0.1), "swdown": _lin(0, 1500, 1),
              "lwdown": _lin(50, 700, 0.5), "precip": _LOG, "wind": _lin(0, 50, 0.02)}


# ----------------------------------------------------------------------------- helpers
def summary(x):
    """Exact summary of plain values."""
    x = np.asarray(x, "float64")
    x = x[np.isfinite(x)]
    out = {"n": len(x)}
    if len(x) == 0:
        return out
    ps = np.percentile(x, PCTS)
    out.update({"min": x.min(), **{f"p{p:g}": v for p, v in zip(PCTS, ps)}, "max": x.max(),
                "mean": x.mean(), "sd": x.std(), "iqr": ps[PCTS.index(75)] - ps[PCTS.index(25)]})
    return out


def hist_summary(h, edges, n, s1, s2, vmin, vmax):
    """Summary of a weighted histogram (quantiles interpolated within bins;
    the open end bins use the exact min and max)."""
    total = h.sum()
    out = {"n": int(n)}
    if total <= 0:
        return out
    e = edges.copy()
    e[0], e[-1] = vmin, vmax
    e = np.clip(e, vmin, vmax)
    cum = np.concatenate([[0.0], np.cumsum(h)]) / total
    qs = {}
    for p in PCTS:
        f = p / 100.0
        i = int(np.searchsorted(cum, f, side="left"))          # first edge with cum >= f
        i = min(max(i, 1), len(cum) - 1)
        c0, c1 = cum[i - 1], cum[i]
        frac = 0.0 if c1 == c0 else (f - c0) / (c1 - c0)
        qs[f"p{p:g}"] = e[i - 1] + frac * (e[i] - e[i - 1])
    mean = s1 / total
    sd = np.sqrt(max(s2 / total - mean ** 2, 0.0))
    out.update({"min": vmin, **qs, "max": vmax, "mean": mean, "sd": sd, "iqr": qs["p75"] - qs["p25"]})
    return out


def counts_outside(x, lo, hi, w=None):
    """(valid, below, above) totals, optionally weighted."""
    ok = np.isfinite(x)
    if w is None:
        return float(ok.sum()), float((x < lo).sum()), float((x > hi).sum())
    w = np.broadcast_to(w, x.shape)
    return float(w[ok].sum()), float(w[x < lo].sum()), float(w[x > hi].sum())


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- towers (group A)
def load_towers(towers_dir, sites):
    """Tower tables -> {quantity: DataFrame(site, time, value)} for hourly and half-hourly."""
    out = {}
    for key, name in (("hourly", "tower_hourly.csv.gz"), ("halfhourly", "tower_halfhourly.csv.gz")):
        t = pd.read_csv(Path(towers_dir) / name)
        t["time"] = pd.to_datetime(t["time"])
        tables = {}
        for q in QUANTITIES:
            v = t[q].values.astype("float64")
            keep = np.isfinite(v)
            tables[q] = pd.DataFrame({"site": t["site"].values[keep], "time": t["time"].values[keep], "value": v[keep]})
        out[key] = tables
    return out["hourly"], out["halfhourly"]


# ----------------------------------------------------------------------------- pass 1: tower cells (B, C)
def era5_files(era5_dir, years):
    files = [Path(era5_dir) / f"era5_hourly_{y}.nc" for y in range(years[0], years[1] + 1)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing ERA5 files: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return files


def tower_cells_series(files, ucells, cache):
    """ERA5 at the unique tower cells, all hours -> Dataset (time, tcell)."""
    if cache.exists():
        ds = xr.open_dataset(cache).load()
        if list(ds.cell.values) == list(ucells) and pd.Timestamp(ds.time.values[0]).year == int(files[0].stem[-4:]) \
                and pd.Timestamp(ds.time.values[-1]).year == int(files[-1].stem[-4:]):
            return ds
        print("cache does not match the cells/years, rebuilding", flush=True)
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds[VARIABLES].isel(cell=list(ucells)).astype("float32").load())
    ds = xr.concat(parts, dim="time")
    ds.to_netcdf(cache, encoding={v: {"zlib": True, "complevel": 1} for v in VARIABLES})
    return ds


# ----------------------------------------------------------------------------- pass 2: all Australia (D)
_G = {}


def _init(cells, edges, thresholds, start_year, n_years):
    _G.update(cells=cells, edges=edges, thr=thresholds, start=start_year, n_years=n_years)


def year_pass(path):
    """One ERA5 year file -> histograms, sums and per-cell counts for every quantity."""
    cells, edges, thr = _G["cells"], _G["edges"], _G["thr"]
    t0 = time.time()
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        data = {v: ds[v].values.astype("float32") for v in VARIABLES}
    file_year = times[0].year
    w = cells.weight.values
    offsets = cells.utc_offset.values
    groups = [(off, np.nonzero(offsets == off)[0]) for off in np.unique(offsets)]
    # local year (relative to file year: 0 or 1), month and hour per offset group
    local = {}
    for off, _ in groups:
        lt = times + pd.Timedelta(hours=float(off))
        local[off] = ((lt.year.values - file_year).astype(np.int64), lt.month.values - 1, lt.hour.values)

    res = {"file_year": file_year}
    for q in QUANTITIES:
        x = data[q]
        e = edges[q]
        nb = len(e) - 1
        h_ym = np.zeros((2, 12, nb))
        h_hr = np.zeros((24, nb))
        for off, idx in groups:
            xg = x[:, idx]
            ok = np.isfinite(xg)
            b = np.searchsorted(e, xg, side="right") - 1                 # e[b] <= x < e[b+1]
            wg = np.broadcast_to(w[idx][None, :], xg.shape)
            yl, ml, hl = local[off]
            ym = (yl * 12 + ml)[:, None]
            inyr = (yl >= 0) & (yl <= 1)
            okym = ok & inyr[:, None]
            comb = (np.broadcast_to(ym, xg.shape)[okym] * nb + b[okym])
            h_ym += np.bincount(comb, weights=wg[okym], minlength=2 * 12 * nb).reshape(2, 12, nb)
            comb = (np.broadcast_to(hl[:, None], xg.shape)[ok] * nb + b[ok])
            h_hr += np.bincount(comb, weights=wg[ok], minlength=24 * nb).reshape(24, nb)
        ok = np.isfinite(x)
        xf = np.where(ok, x, 0).astype("float64")
        wb = np.broadcast_to(w[None, :], x.shape)
        lo, hi = thr[MAIN][q]
        lo_b, hi_b = thr[ERA5_OWN][q]
        res[q] = {
            "h_ym": h_ym, "h_hr": h_hr,
            "n": int(ok.sum()), "s1": float((wb * xf).sum()), "s2": float((wb * xf * xf).sum()),
            "min": float(np.nanmin(x)) if ok.any() else np.inf, "max": float(np.nanmax(x)) if ok.any() else -np.inf,
            "cell_valid": ok.sum(axis=0).astype(np.int32),
            "cell_below": (x < lo).sum(axis=0).astype(np.int32),
            "cell_above": (x > hi).sum(axis=0).astype(np.int32),
            "cell_below_own": (x < lo_b).sum(axis=0).astype(np.int32),
            "cell_above_own": (x > hi_b).sum(axis=0).astype(np.int32),
        }
    res["seconds"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- shares from histograms
def share_from_hist(h, edges, lo, hi):
    """(total, below, above) from a histogram whose edges include lo and
    nextafter(hi): exact."""
    total = h.sum(axis=-1)
    i_lo = np.searchsorted(edges, lo, side="left")
    i_hi = np.searchsorted(edges, np.nextafter(hi, np.inf), side="left")
    if edges[i_lo] != lo or edges[i_hi] != np.nextafter(hi, np.inf):
        raise ValueError("range end is not a bin edge")
    return total, h[..., :i_lo].sum(axis=-1), h[..., i_hi:].sum(axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers", default="data", help="folder with tower_hourly.csv.gz etc.")
    p.add_argument("--era5", required=True, help="folder with era5_hourly_<year>.nc and cells.csv")
    p.add_argument("--years", nargs=2, type=int, default=[1980, 2025])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    sites = pd.read_csv(Path(args.towers) / "tower_sites.csv").set_index("site")
    cells = pd.read_csv(Path(args.era5) / "cells.csv")
    files = era5_files(args.era5, args.years)

    # each tower -> nearest Australian land cell
    d = haversine_km(sites.lat.values[:, None], sites.lon.values[:, None], cells.lat.values[None], cells.lon.values[None])
    sites["cell"] = d.argmin(axis=1)
    sites["cell_km"] = d.min(axis=1)
    sites["cell_lat"] = cells.lat.values[sites.cell]
    sites["cell_lon"] = cells.lon.values[sites.cell]
    sites[["lat", "lon", "cell", "cell_lat", "cell_lon", "cell_km"]].to_csv(out / "sites_cells.csv")
    ucells = sorted(sites.cell.unique())
    print(f"{len(sites)} towers in {len(ucells)} ERA5 cells (max distance {sites.cell_km.max():.1f} km)", flush=True)

    A, A_hh = load_towers(args.towers, sites)
    print(f"towers loaded ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- pass 1: ERA5 at the tower cells
    tc = tower_cells_series(files, ucells, out / "era5_tower_cells.nc")
    tc_times = pd.DatetimeIndex(tc.time.values)
    col = {c: i for i, c in enumerate(tc.cell.values)}
    print(f"pass 1 done: ERA5 at tower cells, {len(tc_times):,} hours ({time.time() - t_start:.0f} s)", flush=True)

    # Matched pairs: for each base variable and site, the tower's measured
    # hours and ERA5 at the site's cell in those same hours.
    pairs = {}             # variable -> list of (site, tower values, era5 values, positions, cell column)
    for v in VARIABLES:
        pairs[v] = []
        for s, g in A[v].groupby("site"):
            pos = tc_times.get_indexer(pd.DatetimeIndex(g["time"].values))
            have = pos >= 0
            c = col[sites.loc[s, "cell"]]
            pairs[v].append((s, g["value"].values[have], tc[v].values[pos[have], c].astype("float64"), pos[have], c))

    pvg_rows = []
    for v in VARIABLES:
        for s, tv, ev, _, _ in pairs[v]:
            ok = np.isfinite(ev)
            tv_ok, ev_ok = tv[ok], ev[ok]
            row = {"site": s, "variable": v, "n": int(ok.sum())}
            if ok.sum() > 2:
                row.update({"tower_mean": tv_ok.mean(), "era5_mean": ev_ok.mean(), "bias": ev_ok.mean() - tv_ok.mean(),
                            "r": np.corrcoef(tv_ok, ev_ok)[0, 1] if tv_ok.std() > 0 and ev_ok.std() > 0 else np.nan,
                            "rmse": np.sqrt(np.mean((ev_ok - tv_ok) ** 2)),
                            "tower_sd": tv_ok.std(), "era5_sd": ev_ok.std(),
                            "sd_ratio": ev_ok.std() / tv_ok.std() if tv_ok.std() > 0 else np.nan,
                            "tower_p0.1": np.percentile(tv_ok, 0.1), "era5_p0.1": np.percentile(ev_ok, 0.1),
                            "tower_p99.9": np.percentile(tv_ok, 99.9), "era5_p99.9": np.percentile(ev_ok, 99.9)})
                if v == "precip":
                    row["tower_wet_share"] = float((tv_ok >= WET_MM_H).mean())
                    row["era5_wet_share"] = float((ev_ok >= WET_MM_H).mean())
            pvg_rows.append(row)
    pd.DataFrame(pvg_rows).to_csv(out / "point_vs_grid.csv", index=False, float_format="%.6g")

    # Groups per variable. A = tower values; B = ERA5 at the hours the tower
    # measured that variable; C = ERA5 at the tower cells, every hour.
    groups = {}
    for q in QUANTITIES:
        c_all = tc[q].values                                          # (time, tcell)
        b = np.concatenate([ev for _, _, ev, _, _ in pairs[q]])
        groups[q] = {"A_all": A[q]["value"].values, "A_hh": A_hh[q]["value"].values,
                     "B": b[np.isfinite(b)], "C": c_all[np.isfinite(c_all)].astype("float64")}

    # ---------------- ranges (thresholds)
    thr = {MAIN: {}, "tower_p1-99": {}, "tower_min-max": {}, "halfhourly_p0.1-99.9": {},
           ERA5_OWN: {}, "towercells_p0.1-99.9": {}}
    for q, g in groups.items():
        a = g["A_all"]
        thr[MAIN][q] = tuple(np.percentile(a, [0.1, 99.9]))
        thr["tower_p1-99"][q] = tuple(np.percentile(a, [1, 99]))
        thr["tower_min-max"][q] = (a.min(), a.max())
        thr["halfhourly_p0.1-99.9"][q] = tuple(np.percentile(g["A_hh"], [0.1, 99.9]))
        thr[ERA5_OWN][q] = tuple(np.percentile(g["B"], [0.1, 99.9]))
        thr["towercells_p0.1-99.9"][q] = tuple(np.percentile(g["C"], [0.1, 99.9]))
    for s in sites.index:
        name = f"without_{s}"
        thr[name] = {}
        for q in QUANTITIES:
            a = A[q]["value"].values[A[q]["site"].values != s]
            thr[name][q] = tuple(np.percentile(a, [0.1, 99.9]))
    # Round every range end to the nearest 32-bit float. The ERA5 values are
    # 32-bit, and many sit exactly on the tower values' 0.01 steps (e.g. 40.41).
    # With 32-bit ends, "x < lo" and "x > hi" give the same answer whether
    # numpy compares in 32 or 64 bits (numpy 1.x compares a float32 array with
    # a float64 number in 32 bits, numpy 2 in 64), and a value equal to an end
    # counts as inside. Changes the ends by at most 1 part in 10 million.
    thr = {r: {q: (float(np.float32(lo)), float(np.float32(hi))) for q, (lo, hi) in d_.items()} for r, d_ in thr.items()}
    pd.DataFrame([{"range": r, "quantity": q, "lo": lo, "hi": hi} for r, d_ in thr.items() for q, (lo, hi) in d_.items()]
                 ).to_csv(out / "thresholds.csv", index=False)   # full precision: the ends are bin edges

    # bin edges: fine grid + every range end (upper ends as the next float up)
    edges = {}
    for q in QUANTITIES:
        ends = [v for d_ in thr.values() for v in (d_[q][0], np.nextafter(d_[q][1], np.inf))]
        # extra edges: the wet threshold, and a sliver above 0 so the many exact
        # zeros (night SW, calm, saturated air) sit in their own bin for quantiles
        extra = [WET_MM_H] if q == "precip" else [0.0, 1e-6]
        edges[q] = np.unique(np.concatenate([[-np.inf], FINE_EDGES[q], ends, extra, [np.inf]]))

    # ---------------- pass 2: every Australian cell, every hour
    years = list(range(args.years[0], args.years[1] + 1))
    ny, nb_of = len(years), {q: len(edges[q]) - 1 for q in QUANTITIES}
    D = {q: {"h_ym": np.zeros((ny + 1, 12, nb_of[q])), "h_hr": np.zeros((24, nb_of[q])), "n": 0, "s1": 0.0, "s2": 0.0,
             "min": np.inf, "max": -np.inf,
             **{k: np.zeros((ny, len(cells)), np.int32) for k in
                ["cell_valid", "cell_below", "cell_above", "cell_below_own", "cell_above_own"]}}
         for q in QUANTITIES}
    init = (cells, edges, thr, args.years[0], ny)
    pool = Pool(args.workers, initializer=_init, initargs=init) if args.workers > 1 else None
    if pool is None:
        _init(*init)
    it = pool.imap_unordered(year_pass, [str(f) for f in files]) if pool else map(year_pass, [str(f) for f in files])
    for res in it:
        yi = res["file_year"] - args.years[0]
        for q in QUANTITIES:
            r, acc = res[q], D[q]
            acc["h_ym"][yi:yi + 2] += r["h_ym"]
            acc["h_hr"] += r["h_hr"]
            for k in ("n", "s1", "s2"):
                acc[k] += r[k]
            acc["min"], acc["max"] = min(acc["min"], r["min"]), max(acc["max"], r["max"])
            for k in ["cell_valid", "cell_below", "cell_above", "cell_below_own", "cell_above_own"]:
                acc[k][yi] = r[k]
        print(f"pass 2: {res['file_year']} done in {res['seconds']:.0f} s", flush=True)
    if pool:
        pool.close()
    for q in QUANTITIES:
        # Whole-period histogram: every hour in the files. The by-year
        # histogram drops the few local hours that fall after the last year
        # (they are in the last UTC file but are 1 January local time).
        D[q]["h_all"] = D[q]["h_ym"].sum(axis=(0, 1))
        D[q]["h_ym"] = D[q]["h_ym"][:ny]
    print(f"pass 2 done ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- ranges.csv
    rows = []
    for q in QUANTITIES:
        g = groups[q]
        for name, vals in (("A_tower", g["A_all"]), ("A_tower_halfhourly", g["A_hh"]), ("B_matched", g["B"]),
                           ("C_tower_cells", g["C"])):
            r = {"group": name, "quantity": q, **summary(vals)}
            rows.append(r)
        hD = D[q]["h_all"]
        rows.append({"group": "D_australia", "quantity": q,
                     **hist_summary(hD, edges[q], D[q]["n"], D[q]["s1"], D[q]["s2"], D[q]["min"], D[q]["max"])})
    ranges = pd.DataFrame(rows)
    # wet share (share of all hours with >= 0.2 mm/h)
    wet = {"A_tower": (groups["precip"]["A_all"] >= WET_MM_H).mean(),
           "A_tower_halfhourly": (groups["precip"]["A_hh"] >= WET_MM_H).mean(),
           "B_matched": (groups["precip"]["B"] >= WET_MM_H).mean(),
           "C_tower_cells": (groups["precip"]["C"] >= WET_MM_H).mean()}
    hP = D["precip"]["h_all"]
    wet["D_australia"] = hP[np.searchsorted(edges["precip"], WET_MM_H):].sum() / hP.sum()
    ranges["wet_share"] = np.where(ranges.quantity == "precip", ranges.group.map(wet), np.nan)
    ranges.to_csv(out / "ranges.csv", index=False, float_format="%.10g")

    # ---------------- shares.csv (A, B, C exact on values; D exact from the histogram)
    rows = []
    for rname, d_ in thr.items():
        if rname.startswith("without_"):
            continue
        for q in QUANTITIES:
            lo, hi = d_[q]
            g = groups[q]
            for gname, vals in (("A_tower", g["A_all"]), ("B_matched", g["B"]), ("C_tower_cells", g["C"])):
                n, below, above = counts_outside(vals, lo, hi)
                rows.append({"range": rname, "quantity": q, "group": gname, "n": n,
                             "below": below / n, "above": above / n, "outside": (below + above) / n})
            tot, below, above = share_from_hist(D[q]["h_all"], edges[q], lo, hi)
            rows.append({"range": rname, "quantity": q, "group": "D_australia", "n": D[q]["n"],
                         "below": below / tot, "above": above / tot, "outside": (below + above) / tot})
    shares = pd.DataFrame(rows)
    shares.to_csv(out / "shares.csv", index=False, float_format="%.10g")

    # ---------------- by year / month / hour, trend
    rows_y, rows_m, rows_h, rows_t = [], [], [], []
    for q in QUANTITIES:
        for rname in (MAIN, ERA5_OWN):
            lo, hi = thr[rname][q]
            tot, below, above = share_from_hist(D[q]["h_ym"].sum(axis=1), edges[q], lo, hi)
            for i, y in enumerate(years):
                rows_y.append({"range": rname, "quantity": q, "year": y, "weight": tot[i],
                               "below": below[i] / tot[i], "above": above[i] / tot[i], "outside": (below[i] + above[i]) / tot[i]})
            ok = tot > 0
            share = (below + above)[ok] / tot[ok]
            yrs = np.array(years)[ok]
            slope = np.polyfit(yrs, share, 1)[0] if ok.sum() > 2 else np.nan
            rows_t.append({"range": rname, "quantity": q, "first_year": yrs.min(), "last_year": yrs.max(),
                           "share_first_decade": share[:10].mean(), "share_last_decade": share[-10:].mean(),
                           "trend_pp_per_decade": slope * 100 * 10,
                           "trend_below_pp_per_decade": np.polyfit(yrs, below[ok] / tot[ok], 1)[0] * 1000,
                           "trend_above_pp_per_decade": np.polyfit(yrs, above[ok] / tot[ok], 1)[0] * 1000})
            tot, below, above = share_from_hist(D[q]["h_ym"].sum(axis=0), edges[q], lo, hi)
            for m in range(12):
                rows_m.append({"range": rname, "quantity": q, "month": m + 1,
                               "below": below[m] / tot[m], "above": above[m] / tot[m], "outside": (below[m] + above[m]) / tot[m]})
            tot, below, above = share_from_hist(D[q]["h_hr"], edges[q], lo, hi)
            for h in range(24):
                if tot[h] > 0:
                    rows_h.append({"range": rname, "quantity": q, "hour": h,
                                   "below": below[h] / tot[h], "above": above[h] / tot[h], "outside": (below[h] + above[h]) / tot[h]})
    pd.DataFrame(rows_y).to_csv(out / "shares_by_year.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_m).to_csv(out / "shares_by_month.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_h).to_csv(out / "shares_by_hour.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_t).to_csv(out / "trend.csv", index=False, float_format="%.6g")

    # ---------------- tower value (leave one site out)
    rows = []
    hD = {q: D[q]["h_all"] for q in QUANTITIES}
    base = {q: share_from_hist(hD[q], edges[q], *thr[MAIN][q]) for q in QUANTITIES}
    for s in sites.index:
        for q in QUANTITIES:
            lo, hi = thr[f"without_{s}"][q]
            tot, below, above = share_from_hist(hD[q], edges[q], lo, hi)
            t0, b0, a0 = base[q]
            rows.append({"site": s, "quantity": q, "hours": int((A[q]["site"] == s).sum()), "lo_without": lo, "hi_without": hi,
                         "outside_with_all": (b0 + a0) / t0, "outside_without": (below + above) / tot,
                         "increase": (below + above) / tot - (b0 + a0) / t0,
                         "increase_below": below / tot - b0 / t0, "increase_above": above / tot - a0 / t0})
    pd.DataFrame(rows).to_csv(out / "tower_value.csv", index=False, float_format="%.10g")

    # ---------------- per_cell.nc and histograms.nc
    pc = xr.Dataset(coords={"year": years, "cell": cells.cell.values})
    pc["lat"] = ("cell", cells.lat.values)
    pc["lon"] = ("cell", cells.lon.values)
    pc["weight"] = ("cell", cells.weight.values)
    for q in QUANTITIES:
        for k in ["valid", "below", "above", "below_own", "above_own"]:
            pc[f"{q}_{k}"] = (("year", "cell"), D[q][f"cell_{k}"])
    pc.attrs.update({"main_range": MAIN, "own_range": ERA5_OWN,
                     "note": "hours per cell and UTC year; below/above = outside the main tower range; *_own = outside ERA5's own range at the tower hours (B)"})
    pc.to_netcdf(out / "per_cell.nc", encoding={v: {"zlib": True} for v in pc.data_vars})

    hs = xr.Dataset(coords={"year": years, "month": np.arange(1, 13), "hour": np.arange(24)})
    for q in QUANTITIES:
        e = edges[q]
        dim = f"bin_{q}"
        hs[f"{q}_edges"] = (f"edge_{q}", e)
        hs[f"{q}_D"] = (dim, D[q]["h_all"])
        hs[f"{q}_D_year_month"] = (("year", "month", dim), D[q]["h_ym"])
        hs[f"{q}_D_hour"] = (("hour", dim), D[q]["h_hr"])
        g = groups[q]
        for gname, vals in (("A", g["A_all"]), ("A_halfhourly", g["A_hh"]), ("B", g["B"]), ("C", g["C"])):
            v = vals[np.isfinite(vals)]
            hs[f"{q}_{gname}"] = (dim, np.bincount(np.searchsorted(e, v, side="right") - 1, minlength=len(e) - 1).astype("float64"))
    hs.attrs["note"] = "bin i holds edges[i] <= x < edges[i+1]; D weighted by cos(lat), A/B/C plain counts"
    hs.to_netcdf(out / "histograms.nc", encoding={v: {"zlib": True} for v in hs.data_vars})

    info = {"years": args.years, "cells": len(cells), "towers": len(sites), "tower_cells": len(ucells),
            "wet_mm_per_h": WET_MM_H, "main_range": MAIN, "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    main_sh = shares[(shares.range == MAIN) & (shares.group == "D_australia")].set_index("quantity")["outside"]
    print("share of Australian land-hours outside the tower range (0.1-99.9 percentiles):")
    for q in QUANTITIES:
        print(f"   {q:11s} {100 * main_sh[q]:6.2f} %")
    print(f"done in {info['minutes']} min -> {out}")


if __name__ == "__main__":
    main()
