"""
coverage_pairs.py
=================

Step 5 (Gadi). Like coverage_stats.py, but for PAIRS of variables: how much of
Australia's weather (ERA5, hourly, 1980-2025) falls outside the combinations
of two variables the towers have measured. All 28 pairs of the 8 variables.

    python3 coverage_pairs.py --towers data --era5 /scratch/dt6/$USER/fc/era5_hourly \
        --single /scratch/dt6/$USER/fc/results --years 1980 2025 --workers 8 \
        --out /scratch/dt6/$USER/fc/results_pairs

--single is the coverage_stats.py results folder: it supplies the 1-variable
ranges (thresholds.csv) and the ERA5-at-tower-cells file (era5_tower_cells.nc),
so the two analyses use exactly the same numbers. It must cover --years.

The tower envelope of a pair (the 2-D version of the 0.1-99.9 % range)
------------------------------------------------------------------------
1. Grid. Each variable's axis is cut into equal bins, 40 bins across its
   tower range (the 1-variable 0.1-99.9 % range), continued with the same width
   one full range beyond each end. Rain uses log10(rain + 0.1 mm/h) as its axis
   (so 0, 0.1, 1, 10, 100 mm/h are evenly spread); all other variables are used
   as they are. Values beyond the grid count as outside.
   Rain is first put on the tower gauges' own steps: tipping buckets record
   0.2 mm (a few 0.1 mm), so tower hourly rain only takes the values 0, 0.2,
   0.4, ... mm/h, while ERA5 rain is continuous. Left alone, ERA5 hours between
   two gauge steps (e.g. 0.05 or 0.3 mm/h) would land in bins no gauge can ever
   fill and count as "never measured". So for the pairs, rain (tower and ERA5
   alike) is rounded to the nearest 0.2 mm/h, as a gauge would record it.
   (The 1-variable ranges and combination-only use the unrounded values.)
2. Count the fully measured tower hours (both variables measured) in each 2-D bin.
3. Drop the most sparsely visited bins, holding together at most 0.2 % of the
   tower hours (the same 0.2 % the 1-variable range drops: 0.1 % at each end).
   Bins are dropped by count level (all bins with 1 hour, then all with 2, ...),
   so there are no arbitrary tie breaks. What is left is the envelope: the
   "highest-density region" holding at least 99.8 % of the tower hours.
4. An ERA5 hour is outside if its bin is not in the envelope.
Also reported: "combination-only" = outside the 2-D envelope although BOTH
variables are inside their own 1-variable tower range. That is the part of the
gap that only shows up when variables are looked at together.
(The envelope is its own definition: an hour just beyond a 1-variable range can
still be inside the envelope if towers often measured that combination.)

Sensitivity variants (bin width x trimmed share): 20 / 40 / 80 bins across the
range x 0 % / 0.2 % / 1 % trimmed. Main = 40 bins, 0.2 %.

Groups as before: A towers, B ERA5 at the towers in the same hours (both
variables measured), C ERA5 at the tower cells every hour, D all of Australia
(cos-latitude weighted).

Exactness: an hour's status depends only on its bin, so each share outside is a
sum of whole bins of a histogram kept on the finest grid (80 bins per range;
the 40 and 20 grids are 2x2 and 4x4 blocks of it), i.e. an exact count. For the
main variant, Australia is also counted hour by hour (needed for the per-year,
per-month, per-hour, per-cell and combination-only numbers) and the script
stops if the two counts disagree.

Outputs (in --out):
    pairs_grid.csv         per variable: axis, tower range, grid start, bin width, bins
    pairs_shares.csv       per pair, variant and group: share outside (+ combination-only for main)
    pairs_by_year.csv      D, main variant: share outside and combination-only by local year
    pairs_by_month.csv     ... by local month
    pairs_by_hour.csv      ... by local hour
    pairs_trend.csv        first/last decade and linear trend of the yearly shares (pp per decade)
    pairs_tower_value.csv  D share outside the main envelope when each site is left out
    pairs_per_cell.nc      per cell and pair: valid hours, outside, combination-only
    pairs_histograms.nc    per pair: fine-grid histograms of A, B, C, D and the main envelope
    pairs_run_info.json
"""

import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from coverage_stats import VARIABLES, MAIN, era5_files, haversine_km

PAIRS = list(itertools.combinations(VARIABLES, 2))       # 28 pairs, fixed order
N_MAIN = 40          # bins across the tower range on the main grid
FINE = 2             # fine bins per main bin (fine grid = 80 bins per range)
EXT = 1.0            # grid continues this many ranges beyond each end
GRIDS = {"bins20": 4, "bins40": 2, "bins80": 1}   # name -> fine bins per bin
TRIMS = {"trim0": 0.0, "trim0.2": 0.002, "trim1": 0.01}
MAIN_VARIANT = ("bins40", "trim0.2")
RAIN_OFFSET = 0.1    # mm/h, in log10(rain + 0.1)
RAIN_STEP = 0.2      # mm/h, tower gauge resolution: rain is rounded to it for the pairs
N_FINE = int(round((1 + 2 * EXT) * N_MAIN * FINE))      # 240 fine bins per axis
assert all(N_FINE % k == 0 for k in GRIDS.values()) and GRIDS[MAIN_VARIANT[0]] == FINE


# ----------------------------------------------------------------------------- grid
def transform(v, x):
    """Axis value of variable v (float64): for rain, log10(r + 0.1) with r = rain rounded
    to the nearest gauge step (0.2 mm/h); other variables as they are."""
    x = np.asarray(x, dtype="float64")
    if v != "precip":
        return x
    r = np.floor(np.clip(x, 0, None) / RAIN_STEP + 0.5) * RAIN_STEP
    return np.log10(r + RAIN_OFFSET)


def make_grid(lo, hi):
    """Fine grid of one variable from its tower range ON THE AXIS SCALE (transformed)."""
    return {"axis_lo": lo, "axis_hi": hi, "start": lo - EXT * (hi - lo),
            "width": (hi - lo) / (N_MAIN * FINE), "n": N_FINE}


def fine_index(t, g):
    """Fine bin of axis values t; -1 = beyond the grid or missing."""
    with np.errstate(invalid="ignore"):
        i = np.floor((t - g["start"]) / g["width"])
        ok = np.isfinite(i) & (i >= 0) & (i < g["n"])
    return np.where(ok, i, -1).astype(np.int32)


def slot_index(ix, iy):
    """Flat fine bin of a pair (ix * N_FINE + iy); N_FINE**2 (the overflow slot)
    if either value is beyond the grid."""
    return np.where((ix >= 0) & (iy >= 0), ix.astype(np.int64) * N_FINE + iy, N_FINE * N_FINE)


def hist(slots, weights=None):
    """Flat fine histogram, last element = overflow (beyond the grid)."""
    return np.bincount(slots, weights=weights, minlength=N_FINE * N_FINE + 1).astype("float64")


def coarsen(h2, k):
    """Sum k x k blocks of a 2-D fine histogram."""
    n = h2.shape[0]
    return h2.reshape(n // k, k, n // k, k).sum(axis=(1, 3))


def envelope(h2, trim):
    """Highest-density region of a 2-D histogram of tower hours: drop every bin
    with count <= c, for the largest count level c whose bins together hold at
    most trim * total hours. Returns a boolean mask of kept bins."""
    total = h2.sum()
    counts = np.sort(h2[h2 > 0])
    if total == 0 or trim <= 0:
        return h2 > 0
    levels, first = np.unique(counts, return_index=True)
    held = np.cumsum(counts)[np.append(first[1:], len(counts)) - 1]   # hours in bins with count <= level
    ok = held <= trim * total
    cut = levels[ok].max() if ok.any() else 0
    return h2 > cut


def expand(mask, k):
    """Coarse-grid mask -> fine-grid mask (each coarse bin becomes k x k)."""
    return np.repeat(np.repeat(mask, k, axis=0), k, axis=1)


def flat_env(mask_fine):
    """Fine-grid mask -> lookup table over flat slots; the overflow slot is outside."""
    return np.append(mask_fine.ravel(), False)


def tower_envelopes(a_flat):
    """All variants' envelopes (fine-grid masks) from a flat tower histogram."""
    a2 = a_flat[:-1].reshape(N_FINE, N_FINE)
    return {(g, t): expand(envelope(coarsen(a2, k), tr), k) for g, k in GRIDS.items() for t, tr in TRIMS.items()}


def in_range(v, x, rng):
    """Inside the 1-variable tower range, compared exactly as coverage_stats.py does
    (outside = x < lo or x > hi; the range ends are float32 numbers)."""
    lo, hi = rng[v]
    with np.errstate(invalid="ignore"):
        return (x >= lo) & (x <= hi)


# ----------------------------------------------------------------------------- pass over all of Australia
_G = {}


def _init(cells, grids, env_main, ranges):
    _G.update(cells=cells, grids=grids, env=env_main, rng=ranges)


def year_pass(path):
    """One ERA5 file: per pair, the fine histogram (all hours), and the main-envelope
    status counted hour by hour by local year-month, local hour and cell."""
    cells, grids, env, rng = _G["cells"], _G["grids"], _G["env"], _G["rng"]
    t0 = time.time()
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        data = {v: ds[v].values.astype("float32").ravel() for v in VARIABLES}    # (time, cell) flattened
    file_year = times[0].year
    nt, nc = len(times), len(cells)
    # local year (0/1 relative to the file year; offsets are positive) * 12 + month, and local hour
    ym = np.empty((nt, nc), np.int8)
    hr = np.empty((nt, nc), np.int8)
    for off in np.unique(cells.utc_offset.values):
        cols = np.nonzero(cells.utc_offset.values == off)[0]
        lt = times + pd.Timedelta(hours=float(off))
        ym[:, cols] = ((lt.year.values - file_year) * 12 + lt.month.values - 1)[:, None]
        hr[:, cols] = lt.hour.values[:, None]
    ym, hr = ym.ravel(), hr.ravel()
    cell = np.tile(np.arange(nc, dtype=np.int32), nt)
    w = cells.weight.values
    idx = {v: fine_index(transform(v, data[v]), grids[v]) for v in VARIABLES}
    inside = {v: in_range(v, data[v], rng) for v in VARIABLES}
    res = {"file_year": file_year}
    for x, y in PAIRS:
        valid = np.nonzero(np.isfinite(data[x]) & np.isfinite(data[y]))[0]
        slots = slot_index(idx[x][valid], idx[y][valid])
        c = cell[valid]
        wv = w[c]
        out = ~env[(x, y)][slots]
        combo = out & inside[x][valid] & inside[y][valid]
        r = {"h": hist(slots, wv)}
        for name, key, size in (("ym", ym, 24), ("hr", hr, 24)):
            k = key[valid].astype(np.int64)
            r[name] = np.stack([np.bincount(k, weights=wv, minlength=size),
                                np.bincount(k, weights=wv * out, minlength=size),
                                np.bincount(k, weights=wv * combo, minlength=size)])
        r["cell"] = np.stack([np.bincount(c, minlength=nc), np.bincount(c[out], minlength=nc),
                              np.bincount(c[combo], minlength=nc)]).astype(np.int64)
        res[(x, y)] = r
    res["seconds"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- shares
def share_outside(h, mask_fine):
    """Share of a flat histogram's weight outside a fine-grid envelope (overflow = outside)."""
    total = h.sum()
    if total <= 0:
        return np.nan, 0.0
    return h[~flat_env(mask_fine)].sum() / total, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers", default="data")
    p.add_argument("--era5", required=True)
    p.add_argument("--single", required=True, help="coverage_stats.py results folder")
    p.add_argument("--years", nargs=2, type=int, default=[1980, 2025])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    single = Path(args.single)

    # 1-variable tower ranges, read exactly (round_trip) so they match coverage_stats.py
    thr = pd.read_csv(single / "thresholds.csv", float_precision="round_trip")
    ranges = {v: tuple(thr[(thr.range == MAIN) & (thr.quantity == v)][["lo", "hi"]].values[0]) for v in VARIABLES}
    grids = {v: make_grid(*transform(v, ranges[v])) for v in VARIABLES}
    pd.DataFrame([{"variable": v, "axis": "log10(mm/h rounded to 0.2 + 0.1)" if v == "precip" else "value",
                   "range_lo": ranges[v][0], "range_hi": ranges[v][1], **grids[v]} for v in VARIABLES]
                 ).to_csv(out / "pairs_grid.csv", index=False, float_format="%.17g")

    sites = pd.read_csv(Path(args.towers) / "tower_sites.csv").set_index("site")
    cells = pd.read_csv(Path(args.era5) / "cells.csv")
    if not (cells.cell.values == np.arange(len(cells))).all():
        raise SystemExit("cells.csv: cell ids must be 0..n-1 in order")
    files = era5_files(args.era5, args.years)
    d = haversine_km(sites.lat.values[:, None], sites.lon.values[:, None], cells.lat.values[None], cells.lon.values[None])
    sites["cell"] = d.argmin(axis=1)

    # ---------------- towers (A) and ERA5 at the towers (B, C)
    tower = pd.read_csv(Path(args.towers) / "tower_hourly.csv.gz")
    tower["time"] = pd.to_datetime(tower["time"])
    tower = tower[tower.site.isin(sites.index)].reset_index(drop=True)
    tc = xr.open_dataset(single / "era5_tower_cells.nc").load()
    tc_times = pd.DatetimeIndex(tc.time.values)
    if tc_times[0].year > args.years[0] or tc_times[-1].year < args.years[1]:
        raise SystemExit("era5_tower_cells.nc does not cover --years; use the coverage_stats.py run for these years")
    keep_t = (tc_times.year >= args.years[0]) & (tc_times.year <= args.years[1])
    tc = tc.isel(time=np.nonzero(keep_t)[0])
    tc_times = pd.DatetimeIndex(tc.time.values)
    # The cache must hold exactly what this Python reads from the ERA5 files (xarray versions
    # differ in how they unpack the int16 values), else B/C and D would not be comparable.
    with xr.open_dataset(files[0]) as ds0:
        n0 = ds0.sizes["time"]
        for v in VARIABLES:
            direct = ds0[v].isel(cell=tc.cell.values).values.astype("float32")
            if not np.array_equal(direct, tc[v].values[:n0], equal_nan=True):
                raise SystemExit(f"era5_tower_cells.nc ({v}) differs from {files[0].name} as read now: "
                                 "remake it (delete it and rerun coverage_stats.py) with this Python")
    col = {c: i for i, c in enumerate(tc.cell.values)}
    missing = set(sites.cell) - set(col)
    if missing:
        raise SystemExit(f"era5_tower_cells.nc lacks tower cells {sorted(missing)}")
    pos = tc_times.get_indexer(pd.DatetimeIndex(tower["time"]))
    have = pos >= 0
    tcol = tower["site"].map(lambda s: col[sites.loc[s, "cell"]]).values
    A_val, B_val, C_val = {}, {}, {}
    for v in VARIABLES:
        A_val[v] = tower[v].values.astype("float64")
        b = np.full(len(tower), np.nan, dtype="float32")
        b[have] = tc[v].values[pos[have], tcol[have]]
        B_val[v] = b
        C_val[v] = tc[v].values.astype("float32").ravel()
    A_idx = {v: fine_index(transform(v, A_val[v]), grids[v]) for v in VARIABLES}
    B_idx = {v: fine_index(transform(v, B_val[v]), grids[v]) for v in VARIABLES}
    C_idx = {v: fine_index(transform(v, C_val[v]), grids[v]) for v in VARIABLES}
    for v in VARIABLES:
        beyond = np.isfinite(A_val[v]) & (A_idx[v] < 0)
        if beyond.any():
            raise SystemExit(f"{v}: {beyond.sum()} tower hours beyond the grid - widen EXT")
    print(f"{len(tower):,} tower hours, {len(col)} tower cells x {len(tc_times):,} hours "
          f"({time.time() - t_start:.0f} s)", flush=True)

    # Envelopes and the A, B, C histograms (+ combination-only), per pair
    H, env, combo_abc, A_sites = {}, {}, {}, {}
    for x, y in PAIRS:
        pr = (x, y)
        H[pr], combo_abc[pr] = {}, {}
        a_ok = np.isfinite(A_val[x]) & np.isfinite(A_val[y])
        H[pr]["A"] = hist(slot_index(A_idx[x][a_ok], A_idx[y][a_ok]))
        env[pr] = tower_envelopes(H[pr]["A"])
        b_ok = a_ok & np.isfinite(B_val[x]) & np.isfinite(B_val[y])
        c_ok = np.isfinite(C_val[x]) & np.isfinite(C_val[y])
        main_flat = flat_env(env[pr][MAIN_VARIANT])
        for g, vals, ids, ok in (("A_tower", A_val, A_idx, a_ok), ("B_matched", B_val, B_idx, b_ok),
                                 ("C_tower_cells", C_val, C_idx, c_ok)):
            slots = slot_index(ids[x][ok], ids[y][ok])
            if g != "A_tower":
                H[pr][g[0]] = hist(slots)
            outside = ~main_flat[slots]
            combo = outside & in_range(x, vals[x][ok], ranges) & in_range(y, vals[y][ok], ranges)
            combo_abc[pr][g] = (outside.mean(), combo.mean()) if len(slots) else (np.nan, np.nan)
        # per-site tower histograms (leave one site out)
        site_of = tower["site"].values[a_ok]
        slots = slot_index(A_idx[x][a_ok], A_idx[y][a_ok])
        A_sites[pr] = {s: hist(slots[site_of == s]) for s in np.unique(site_of)}
        kept = H[pr]["A"][flat_env(env[pr][MAIN_VARIANT])].sum() / H[pr]["A"].sum()
        assert kept >= 1 - TRIMS[MAIN_VARIANT[1]] - 1e-12, (pr, kept)
    print(f"envelopes built ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- every Australian cell, every hour
    years = list(range(args.years[0], args.years[1] + 1))
    n_y = len(years)
    D = {pr: {"h": np.zeros(N_FINE * N_FINE + 1), "ym": np.zeros((3, (n_y + 1) * 12)),
              "hr": np.zeros((3, 24)), "cell": np.zeros((3, len(cells)), np.int64)} for pr in PAIRS}
    env_main = {pr: flat_env(env[pr][MAIN_VARIANT]) for pr in PAIRS}
    init = (cells, grids, env_main, ranges)
    pool = Pool(args.workers, initializer=_init, initargs=init) if args.workers > 1 else None
    if pool is None:
        _init(*init)
    paths = [str(f) for f in files]
    for res in (pool.imap_unordered(year_pass, paths) if pool else map(year_pass, paths)):
        yi = res["file_year"] - args.years[0]
        for pr in PAIRS:
            r, acc = res[pr], D[pr]
            acc["h"] += r["h"]
            acc["ym"][:, yi * 12:(yi + 2) * 12] += r["ym"]
            acc["hr"] += r["hr"]
            acc["cell"] += r["cell"]
        print(f"{res['file_year']} done in {res['seconds']:.0f} s", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"all years done ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- shares
    rows, rows_y, rows_m, rows_h, rows_t, rows_v = [], [], [], [], [], []
    for x, y in PAIRS:
        pr, name = (x, y), f"{x}-{y}"
        hD, ymD, hrD = D[pr]["h"], D[pr]["ym"], D[pr]["hr"]
        # the hour-by-hour count must equal the histogram count (main variant)
        sh_main, n_main = share_outside(hD, env[pr][MAIN_VARIANT])
        if not (abs(ymD[0].sum() - n_main) <= 1e-9 * n_main and abs(ymD[1].sum() / ymD[0].sum() - sh_main) <= 1e-9):
            raise SystemExit(f"{name}: hour-by-hour and histogram counts disagree")
        for variant, mask in env[pr].items():
            for g, h in (("A_tower", H[pr]["A"]), ("B_matched", H[pr]["B"]), ("C_tower_cells", H[pr]["C"]),
                         ("D_australia", hD)):
                sh, n = share_outside(h, mask)
                row = {"pair": name, "x": x, "y": y, "grid": variant[0], "trim": variant[1], "group": g,
                       "n": n, "outside": sh, "combination_only": np.nan}
                if variant == MAIN_VARIANT:
                    row["combination_only"] = (ymD[2].sum() / ymD[0].sum() if g == "D_australia"
                                               else combo_abc[pr][g][1])
                rows.append(row)
        # by local year / month / hour (main variant); the local year after the last is a few hours, left out
        ymr = ymD[:, :n_y * 12].reshape(3, n_y, 12)
        tot_y, out_y, com_y = ymr[0].sum(1), ymr[1].sum(1), ymr[2].sum(1)
        for i, yr in enumerate(years):
            if tot_y[i] > 0:
                rows_y.append({"pair": name, "year": yr, "hours_weight": tot_y[i],
                               "outside": out_y[i] / tot_y[i], "combination_only": com_y[i] / tot_y[i]})
        tot_m, out_m, com_m = ymr[0].sum(0), ymr[1].sum(0), ymr[2].sum(0)
        for m in range(12):
            if tot_m[m] > 0:
                rows_m.append({"pair": name, "month": m + 1, "outside": out_m[m] / tot_m[m],
                               "combination_only": com_m[m] / tot_m[m]})
        for h_ in range(24):
            if hrD[0, h_] > 0:
                rows_h.append({"pair": name, "hour": h_, "outside": hrD[1, h_] / hrD[0, h_],
                               "combination_only": hrD[2, h_] / hrD[0, h_]})
        ok = tot_y > 0
        yrs = np.array(years)[ok]
        so, sc = out_y[ok] / tot_y[ok], com_y[ok] / tot_y[ok]
        fit = ok.sum() > 2
        rows_t.append({"pair": name, "years": f"{yrs[0]}-{yrs[-1]}",
                       "outside_first_decade": so[:10].mean(), "outside_last_decade": so[-10:].mean(),
                       "trend_outside_pp_per_decade": np.polyfit(yrs, so, 1)[0] * 1000 if fit else np.nan,
                       "combo_first_decade": sc[:10].mean(), "combo_last_decade": sc[-10:].mean(),
                       "trend_combo_pp_per_decade": np.polyfit(yrs, sc, 1)[0] * 1000 if fit else np.nan})
        # leave one site out (main variant)
        k = GRIDS[MAIN_VARIANT[0]]
        for s, hs in A_sites[pr].items():
            a2 = (H[pr]["A"] - hs)[:-1].reshape(N_FINE, N_FINE)
            sh, _ = share_outside(hD, expand(envelope(coarsen(a2, k), TRIMS[MAIN_VARIANT[1]]), k))
            rows_v.append({"pair": name, "site": s, "hours": int(hs.sum()), "outside_with_all": sh_main,
                           "outside_without": sh, "increase": sh - sh_main})
    shares = pd.DataFrame(rows)
    for df, f in ((shares, "pairs_shares.csv"), (pd.DataFrame(rows_y), "pairs_by_year.csv"),
                  (pd.DataFrame(rows_m), "pairs_by_month.csv"), (pd.DataFrame(rows_h), "pairs_by_hour.csv"),
                  (pd.DataFrame(rows_t), "pairs_trend.csv"), (pd.DataFrame(rows_v), "pairs_tower_value.csv")):
        df.to_csv(out / f, index=False, float_format="%.10g")

    # ---------------- per cell and histograms
    pc = xr.Dataset(coords={"cell": cells.cell.values})
    for c in ("lat", "lon", "weight"):
        pc[c] = ("cell", cells[c].values)
    for x, y in PAIRS:
        for i, k in enumerate(("valid", "out", "combo")):
            pc[f"{x}-{y}_{k}"] = ("cell", D[(x, y)]["cell"][i].astype(np.int32))
    pc.attrs["note"] = ("hours per cell, all years: valid (both variables), outside the main 2-D envelope, "
                        "combination-only (outside the envelope, inside both 1-variable ranges)")
    pc.to_netcdf(out / "pairs_per_cell.nc", encoding={v: {"zlib": True} for v in pc.data_vars})

    hs = xr.Dataset(coords={f"b_{v}": grids[v]["start"] + (np.arange(N_FINE) + 0.5) * grids[v]["width"]
                            for v in VARIABLES})
    for x, y in PAIRS:
        for g, h in (("A", H[(x, y)]["A"]), ("B", H[(x, y)]["B"]), ("C", H[(x, y)]["C"]), ("D", D[(x, y)]["h"])):
            hs[f"{x}-{y}_{g}"] = ((f"b_{x}", f"b_{y}"), h[:-1].reshape(N_FINE, N_FINE))
            hs[f"{x}-{y}_{g}_beyond"] = h[-1]
        hs[f"{x}-{y}_envelope"] = ((f"b_{x}", f"b_{y}"), env[(x, y)][MAIN_VARIANT].astype(np.int8))
    hs.attrs["note"] = ("fine grid (see pairs_grid.csv); coords are bin centres on the axis scale "
                        "(rain: log10(mm/h rounded to 0.2 + 0.1)); A, B, C hour counts, D cos-latitude weighted; "
                        "*_beyond = beyond the grid; envelope = main variant (40 bins, 0.2 %)")
    hs.to_netcdf(out / "pairs_histograms.nc", encoding={v: {"zlib": True} for v in hs.data_vars})

    info = {"years": args.years, "pairs": len(PAIRS), "n_main_bins": N_MAIN, "fine_per_main": FINE, "ext": EXT,
            "trims": TRIMS, "grids": GRIDS, "main": list(MAIN_VARIANT), "rain_offset": RAIN_OFFSET, "rain_step": RAIN_STEP,
            "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "pairs_run_info.json").write_text(json.dumps(info, indent=2))
    m = shares[(shares.grid == MAIN_VARIANT[0]) & (shares.trim == MAIN_VARIANT[1]) & (shares.group == "D_australia")]
    print("share of Australian land-hours outside the tower envelope (main) | combination-only:")
    for r in m.itertuples():
        print(f"   {r.x:7s} x {r.y:7s} {100 * r.outside:6.2f} % | {100 * r.combination_only:6.2f} %")
    print(f"done in {info['minutes']} min -> {out}")


if __name__ == "__main__":
    main()
