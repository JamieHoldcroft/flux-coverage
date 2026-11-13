"""
coverage_aoa.py
===============

Step 6 (Gadi). All variables together: how much of Australia's weather (ERA5,
hourly, 1980-2025) is a combination of conditions unlike any hour the towers
have measured. Two methods, from one nearest-neighbour search:

  AOA (main): the Area of Applicability of Meyer & Pebesma (2021, Methods in
      Ecology and Evolution 12:1620), computed the way their R package CAST does.
      1. Each variable is standardised with the tower mean and sd (all equal weight).
      2. DI of an hour = distance to the nearest tower hour / d_bar, where d_bar is
         the mean distance between two tower hours (estimated from 20 million
         random pairs; it cancels out of the in/out decision, it only scales DI).
      3. Cut-off: every tower hour gets its DI with its own site left out
         (leave one tower out = CAST's spatial cross-validation folds). The cut-off
         is the upper whisker of the box plot of those DIs (R boxplot.stats: the
         largest DI <= upper hinge + 1.5 x hinge spread).
      4. An ERA5 hour is outside (uncovered) if its DI > cut-off.
  kNN-10 (check): distance to the 10th nearest tower hour, so a combination only
      counts as covered if at least 10 tower hours were measured near it. Cut-off =
      99.8th percentile of the towers' own 10th-neighbour distance (own hour left
      out), i.e. the same 0.2 % the 1-variable range trims. Reported in DI units.

    python3 coverage_aoa.py --towers data --era5 /scratch/dt6/$USER/fc/era5_hourly \
        --single /scratch/dt6/$USER/fc/results --pairs /scratch/dt6/$USER/fc/results_pairs \
        --years 1980 2025 --workers 48 --out /scratch/dt6/$USER/fc/results_aoa

Variables (--vars): default tair vpd swdown lwdown precip wind. Qair is left out
(it is calculated from Tair and VPD) and so is surface pressure (it mostly labels
elevation). Any set can be given for a sensitivity run. Rain is used as
log10(rain rounded to 0.2 mm/h + 0.1), exactly as in coverage_pairs.py.
Only hours with every variable measured are used (towers) / valid (ERA5).

--single (coverage_stats.py results) supplies the 1-variable ranges and the
ERA5-at-tower-cells cache. --pairs (coverage_pairs.py results, optional)
supplies the pair envelopes for the breakdown of the uncovered hours into
  single     at least one variable outside its own 1-variable tower range
  pair       all variables inside, but at least one pair outside its envelope
  joint_only every variable and every pair inside: only the full combination is new

Groups as before: A towers (A_loto = own site left out, A_all = own hour left
out), B ERA5 at the towers in the same hours, C ERA5 at the tower cells every
hour, D all of Australia (cos-latitude weighted).

Tower value (--value-every N, default 200): a fixed random 1-in-N sample of the
Australian cell-hours (seeded, so reruns take the same hours) is kept; for each site the AOA share of the sample is
recomputed with that site removed from the towers (cut-off and d_bar fixed).

Outputs (in --out):
    aoa_settings.csv      variables, tower mean and sd per variable
    aoa_threshold.csv     d_bar, LOTO DI quartiles/hinges, AOA cut-off, kNN cut-off, n
    aoa_shares.csv        per group: n, share outside AOA, share outside kNN-10, breakdown, mean DI
    aoa_tower_sites.csv   per site: hours, median LOTO DI, share of its hours outside when left out
    aoa_curve.csv         share of each group above DI = x (threshold curve), AOA and kNN distance
    aoa_by_year.csv       D by local year (AOA, kNN, breakdown)
    aoa_by_month.csv      D by local month (years in --years; the few local hours after the last are left out)
    aoa_by_hour.csv       D by local hour (all hours)
    aoa_trend.csv         first/last decade and linear trend (pp per decade)
    aoa_tower_value.csv   per site: sample share outside with all towers / without the site
    aoa_per_cell.nc       per cell: valid hours, outside AOA, outside kNN, joint-only, mean DI
    aoa_run_info.json
    (--save-di: di_<year>.npy and di_groups.npz, every hour's DI, kNN distance (and category), for
     check_coverage_aoa.py; tests only, large)
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from coverage_stats import VARIABLES as ALL_VARS, MAIN, era5_files, haversine_km
from coverage_pairs import transform, fine_index, slot_index, N_FINE

DEFAULT_VARS = ["tair", "vpd", "swdown", "lwdown", "precip", "wind"]
K_NN = 10            # neighbours for the check
KNN_PCT = 99.8       # tower percentile of the 10th-neighbour distance = kNN cut-off
WHISKER = 1.5        # box-plot whisker coefficient (R default, CAST)
N_DBAR = 20_000_000  # random tower-hour pairs for d_bar
SEED = 20261113
DI_STEP, DI_MAX = 0.001, 3.0    # DI histogram bins (curve only; shares are counted hour by hour)
N_BINS = int(round(DI_MAX / DI_STEP))
CHUNK = 2_000_000    # ERA5 points per tree query
CATS = ["single", "pair", "joint_only"]


# ----------------------------------------------------------------------------- method
def to_space(vals, vars_, mean, sd):
    """{var: values} -> standardised (n, p) float64 matrix (rain transformed first)."""
    return np.column_stack([(transform(v, vals[v]) - mean[i]) / sd[i] for i, v in enumerate(vars_)])


def upper_whisker(x, coef=WHISKER):
    """R boxplot.stats(x)$stats[5]: hinges from fivenum, whisker = largest x <= upper hinge + coef * spread."""
    x = np.sort(np.asarray(x, dtype="float64"))
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1           # 0-based positions
    five = 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])
    lim = five[3] + coef * (five[3] - five[1])
    return x[x <= lim].max(), five


def mean_pair_distance(X, n_pairs, seed=SEED):
    """d_bar: mean Euclidean distance between two different tower hours (random pairs)."""
    rng = np.random.default_rng(seed)
    total, done = 0.0, 0
    while done < n_pairs:
        m = min(5_000_000, n_pairs - done)
        i = rng.integers(0, len(X), m)
        j = rng.integers(0, len(X) - 1, m)
        j = j + (j >= i)                                              # j != i, uniform over the others
        total += np.sqrt(((X[i] - X[j]) ** 2).sum(1)).sum()
        done += m
    return total / n_pairs


def loto_di(X, site, dbar, workers):
    """DI of each tower hour to the nearest hour of ANOTHER site."""
    di = np.empty(len(X))
    for s in np.unique(site):
        own = site == s
        di[own] = cKDTree(X[~own]).query(X[own], k=1, workers=workers)[0] / dbar
    return di


def query(tree, X, workers):
    """(DI numerator d1, d10) of each row, in chunks."""
    d1, d10 = np.empty(len(X)), np.empty(len(X))
    for s in range(0, len(X), CHUNK):
        d, _ = tree.query(X[s:s + CHUNK], k=K_NN, workers=workers)
        d1[s:s + CHUNK], d10[s:s + CHUNK] = d[:, 0], d[:, K_NN - 1]
    return d1, d10


# ----------------------------------------------------------------------------- breakdown of uncovered hours
def load_pair_envelopes(pairs_dir, vars_):
    """Pair envelopes (fine-grid lookup tables) + grids from coverage_pairs.py results."""
    g = pd.read_csv(Path(pairs_dir) / "pairs_grid.csv", float_precision="round_trip").set_index("variable")
    grids = {v: {"start": g.loc[v, "start"], "width": g.loc[v, "width"], "n": int(g.loc[v, "n"])} for v in vars_}
    assert all(gr["n"] == N_FINE for gr in grids.values())
    env = {}
    with xr.open_dataset(Path(pairs_dir) / "pairs_histograms.nc") as hs:
        for x, y in itertools.combinations([v for v in ALL_VARS if v in vars_], 2):   # coverage_pairs order
            env[(x, y)] = np.append(hs[f"{x}-{y}_envelope"].values.astype(bool).ravel(), False)
    return grids, env


def breakdown(vals, ranges, pgrids, penv):
    """Per hour: 0 = some variable outside its 1-D range, 1 = all inside but some pair
    outside its envelope, 2 = every variable and pair inside (joint_only).
    Without pair envelopes, 1 never occurs and 2 means 'all variables inside'."""
    cat = np.full(len(next(iter(vals.values()))), 2, np.int8)
    single = np.zeros(cat.shape, bool)
    for v, x in vals.items():
        lo, hi = ranges[v]
        with np.errstate(invalid="ignore"):
            single |= (x < lo) | (x > hi)
    if penv:
        idx = {v: fine_index(transform(v, x), pgrids[v]) for v, x in vals.items()}
        pair = np.zeros(cat.shape, bool)
        for (x, y), lut in penv.items():
            pair |= ~lut[slot_index(idx[x], idx[y])]
        cat[pair] = 1
    cat[single] = 0
    return cat


# ----------------------------------------------------------------------------- one ERA5 year
def local_keys(times, utc_offset, file_year):
    """(time, cell) local year-month index (0..23, relative to the file year) and local hour."""
    nt, nc = len(times), len(utc_offset)
    ym, hr = np.empty((nt, nc), np.int8), np.empty((nt, nc), np.int8)
    for off in np.unique(utc_offset):
        cols = np.nonzero(utc_offset == off)[0]
        lt = times + pd.Timedelta(hours=float(off))
        ym[:, cols] = ((lt.year.values - file_year) * 12 + lt.month.values - 1)[:, None]
        hr[:, cols] = lt.hour.values[:, None]
    return ym.ravel(), hr.ravel()


def year_pass(path, S, tree, cut, workers, value_every, save_dir):
    t0 = time.time()
    vars_, cells = S["vars"], S["cells"]
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        raw = {v: ds[v].values.astype("float32").ravel() for v in vars_}      # (time, cell) flattened
    nt, nc = len(times), len(cells)
    file_year = times[0].year
    ym, hr = local_keys(times, cells.utc_offset.values, file_year)
    cell = np.tile(np.arange(nc, dtype=np.int64), nt)
    valid = np.ones(nt * nc, bool)
    for v in vars_:
        valid &= np.isfinite(raw[v])
    vi = np.nonzero(valid)[0]
    vals = {v: raw[v][vi] for v in vars_}
    del raw
    X = to_space(vals, vars_, S["mean"], S["sd"])
    d1, d10 = query(tree, X, workers)
    di, dk = d1 / S["dbar"], d10 / S["dbar"]
    out_a, out_k = di > cut["aoa"], dk > cut["knn"]
    cat = breakdown(vals, S["ranges"], S["pgrids"], S["penv"])
    c, w = cell[vi], cells.weight.values[cell[vi]]
    cols = [w, w * out_a, w * out_k] + [w * (out_a & (cat == i)) for i in range(3)] + [w * di]
    res = {"file_year": file_year, "n": len(vi), "n_all": nt * nc}
    for name, key in (("ym", ym[vi].astype(np.int64)), ("hr", hr[vi].astype(np.int64))):
        res[name] = np.stack([np.bincount(key, weights=x, minlength=24) for x in cols])
    res["cell"] = np.stack([np.bincount(c, minlength=nc), np.bincount(c, weights=out_a, minlength=nc),
                            np.bincount(c, weights=out_k, minlength=nc),
                            np.bincount(c, weights=out_a & (cat == 2), minlength=nc),
                            np.bincount(c, weights=di, minlength=nc)])
    res["h_di"] = np.bincount(np.minimum(di / DI_STEP, N_BINS).astype(np.int64), weights=w, minlength=N_BINS + 1)
    res["h_dk"] = np.bincount(np.minimum(dk / DI_STEP, N_BINS).astype(np.int64), weights=w, minlength=N_BINS + 1)
    if value_every:
        # fixed random sample (seeded by year, so reruns take the same cell-hours), 1 in N
        take = np.random.default_rng(SEED + file_year).random(nt * nc)[vi] < 1.0 / value_every
        res["sample"] = (X[take].astype("float32"), w[take], out_a[take])
    if save_dir is not None:
        full = np.full((2, nt * nc), np.nan)
        full[0, vi], full[1, vi] = di, dk
        np.save(Path(save_dir) / f"di_{file_year}.npy", full)
    res["seconds"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers", default="data")
    p.add_argument("--era5", required=True)
    p.add_argument("--single", required=True, help="coverage_stats.py results folder")
    p.add_argument("--pairs", default=None, help="coverage_pairs.py results folder (for the breakdown)")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    p.add_argument("--years", nargs=2, type=int, default=[1980, 2025])
    p.add_argument("--workers", type=int, default=1, help="threads for the neighbour search")
    p.add_argument("--value-every", type=int, default=200, help="tower-value sample: every N-th cell-hour (0 = off)")
    p.add_argument("--save-di", action="store_true", help="save every hour's DI (tests only: large)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vars_ = [v for v in ALL_VARS if v in args.vars]
    if len(vars_) != len(args.vars) or len(vars_) < 2:
        raise SystemExit(f"--vars must be 2+ of {ALL_VARS}")
    single = Path(args.single)
    thr = pd.read_csv(single / "thresholds.csv", float_precision="round_trip")
    ranges = {v: tuple(thr[(thr.range == MAIN) & (thr.quantity == v)][["lo", "hi"]].values[0]) for v in vars_}
    pgrids, penv = load_pair_envelopes(args.pairs, vars_) if args.pairs else (None, None)

    sites = pd.read_csv(Path(args.towers) / "tower_sites.csv").set_index("site")
    cells = pd.read_csv(Path(args.era5) / "cells.csv")
    if not (cells.cell.values == np.arange(len(cells))).all():
        raise SystemExit("cells.csv: cell ids must be 0..n-1 in order")
    files = era5_files(args.era5, args.years)
    d = haversine_km(sites.lat.values[:, None], sites.lon.values[:, None], cells.lat.values[None], cells.lon.values[None])
    sites["cell"] = d.argmin(axis=1)

    # ---------------- towers (A): every hour with all variables measured
    tower = pd.read_csv(Path(args.towers) / "tower_hourly.csv.gz")
    tower["time"] = pd.to_datetime(tower["time"])
    tower = tower[tower.site.isin(sites.index)]
    tower = tower[np.isfinite(tower[vars_].values.astype("float64")).all(1)].reset_index(drop=True)
    A_vals = {v: tower[v].values.astype("float64") for v in vars_}
    T = np.column_stack([transform(v, A_vals[v]) for v in vars_])
    mean, sd = T.mean(0), T.std(0, ddof=1)
    XA = (T - mean) / sd
    site = tower["site"].values
    n_sites = len(np.unique(site))
    print(f"{len(tower):,} tower hours with all of {vars_} at {n_sites} sites", flush=True)

    dbar = mean_pair_distance(XA, N_DBAR)
    di_loto = loto_di(XA, site, dbar, args.workers)
    cut_aoa, five = upper_whisker(di_loto)
    tree = cKDTree(XA)
    dA, _ = tree.query(XA, k=K_NN + 1, workers=args.workers)        # column 0 = the hour itself (or an identical one)
    di_all, dk_all = dA[:, 1] / dbar, dA[:, K_NN] / dbar
    cut_knn = np.percentile(dk_all, KNN_PCT)
    cut = {"aoa": cut_aoa, "knn": cut_knn}
    print(f"d_bar {dbar:.4f}; AOA cut-off DI {cut_aoa:.4f} ({100 * (di_loto > cut_aoa).mean():.2f} % of tower hours "
          f"above when their site is left out); kNN-10 cut-off {cut_knn:.4f} ({time.time() - t_start:.0f} s)", flush=True)

    pd.DataFrame({"variable": vars_, "axis": ["log10(mm/h rounded to 0.2 + 0.1)" if v == "precip" else "value" for v in vars_],
                  "tower_mean": mean, "tower_sd": sd}).to_csv(out / "aoa_settings.csv", index=False, float_format="%.17g")
    pd.DataFrame([{"variables": " ".join(vars_), "tower_hours": len(XA), "sites": n_sites, "d_bar": dbar,
                   "n_pairs_dbar": N_DBAR, "loto_min": five[0], "loto_lower_hinge": five[1], "loto_median": five[2],
                   "loto_upper_hinge": five[3], "loto_max": five[4], "whisker_coef": WHISKER,
                   "aoa_cutoff": cut_aoa, "knn_k": K_NN, "knn_pct": KNN_PCT, "knn_cutoff": cut_knn}]
                 ).to_csv(out / "aoa_threshold.csv", index=False, float_format="%.17g")
    rows_s = []
    for s in np.unique(site):
        own = site == s
        rows_s.append({"site": s, "hours": int(own.sum()), "median_di_loto": np.median(di_loto[own]),
                       "outside_aoa_loto": (di_loto[own] > cut_aoa).mean(),
                       "outside_aoa_all": (di_all[own] > cut_aoa).mean(),
                       "outside_knn_all": (dk_all[own] > cut_knn).mean()})
    pd.DataFrame(rows_s).to_csv(out / "aoa_tower_sites.csv", index=False, float_format="%.10g")

    # ---------------- ERA5 at the towers (B: same hours; C: tower cells, all hours)
    tc = xr.open_dataset(single / "era5_tower_cells.nc").load()
    tc_times = pd.DatetimeIndex(tc.time.values)
    if tc_times[0].year > args.years[0] or tc_times[-1].year < args.years[1]:
        raise SystemExit("era5_tower_cells.nc does not cover --years")
    tc = tc.isel(time=np.nonzero((tc_times.year >= args.years[0]) & (tc_times.year <= args.years[1]))[0])
    tc_times = pd.DatetimeIndex(tc.time.values)
    with xr.open_dataset(files[0]) as ds0:          # the cache must equal what this Python reads (xarray versions)
        n0 = ds0.sizes["time"]
        for v in vars_:
            if not np.array_equal(ds0[v].isel(cell=tc.cell.values).values.astype("float32"), tc[v].values[:n0],
                                  equal_nan=True):
                raise SystemExit(f"era5_tower_cells.nc ({v}) differs from {files[0].name} as read now: remake it")
    col = {c: i for i, c in enumerate(tc.cell.values)}
    if set(sites.cell) - set(col):
        raise SystemExit("era5_tower_cells.nc lacks some tower cells")
    pos = tc_times.get_indexer(pd.DatetimeIndex(tower["time"]))
    tcol = tower["site"].map(lambda s: col[sites.loc[s, "cell"]]).values
    have = pos >= 0
    B_vals = {}
    for v in vars_:
        b = np.full(len(tower), np.nan, dtype="float32")
        b[have] = tc[v].values[pos[have], tcol[have]]
        B_vals[v] = b
    C_vals = {v: tc[v].values.astype("float32").ravel() for v in vars_}
    groups = {}
    for g, vals in (("B_matched", B_vals), ("C_tower_cells", C_vals)):
        ok = np.ones(len(vals[vars_[0]]), bool)
        for v in vars_:
            ok &= np.isfinite(vals[v])
        sub = {v: vals[v][ok] for v in vars_}
        d1, d10 = query(tree, to_space(sub, vars_, mean, sd), args.workers)
        groups[g] = (d1 / dbar, d10 / dbar, breakdown(sub, ranges, pgrids, penv))
    A_cat = breakdown(A_vals, ranges, pgrids, penv)
    groups = {"A_tower_loto": (di_loto, dk_all, A_cat), "A_tower_all": (di_all, dk_all, A_cat), **groups}
    if args.save_di:
        np.savez(out / "di_groups.npz", **{f"{g}_{k}": a for g, t in groups.items() for k, a in zip(("di", "dk", "cat"), t)})
    print(f"groups A-C done ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- every Australian cell, every hour (D)
    S = {"vars": vars_, "cells": cells, "mean": mean, "sd": sd, "dbar": dbar, "ranges": ranges,
         "pgrids": pgrids, "penv": penv}
    years = list(range(args.years[0], args.years[1] + 1))
    n_y = len(years)
    ymD, hrD = np.zeros((7, (n_y + 1) * 12)), np.zeros((7, 24))
    cellD = np.zeros((5, len(cells)))
    hD = {"di": np.zeros(N_BINS + 1), "dk": np.zeros(N_BINS + 1)}
    samples = []
    for f in files:
        r = year_pass(f, S, tree, cut, args.workers, args.value_every, out if args.save_di else None)
        yi = r["file_year"] - args.years[0]
        ymD[:, yi * 12:(yi + 2) * 12] += r["ym"]
        hrD += r["hr"]
        cellD += r["cell"]
        hD["di"] += r["h_di"]
        hD["dk"] += r["h_dk"]
        if args.value_every:
            samples.append(r["sample"])
        print(f"{r['file_year']} done in {r['seconds']:.0f} s ({r['n']:,} of {r['n_all']:,} cell-hours valid)", flush=True)
    print(f"all years done ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- shares
    rows = []
    for g, (di, dk, cat) in groups.items():
        oa = di > cut_aoa
        row = {"group": g, "n": len(di), "outside_aoa": oa.mean(), "outside_knn": (dk > cut_knn).mean(),
               "mean_di": di.mean()}
        for i, cname in enumerate(CATS):
            row[f"outside_{cname}"] = (oa & (cat == i)).mean()
        rows.append(row)
    tot = ymD[0].sum()
    rowD = {"group": "D_australia", "n": int(cellD[0].sum()), "outside_aoa": ymD[1].sum() / tot,
           "outside_knn": ymD[2].sum() / tot, "mean_di": ymD[6].sum() / tot}
    for i, cname in enumerate(CATS):
        rowD[f"outside_{cname}"] = ymD[3 + i].sum() / tot
    rows.append(rowD)
    shares = pd.DataFrame(rows)
    if not args.pairs:
        shares["outside_joint_only"] = shares["outside_pair"] + shares["outside_joint_only"]
        shares["outside_pair"] = np.nan
    shares.to_csv(out / "aoa_shares.csv", index=False, float_format="%.10g")
    # the curve from the D histogram must agree with the hour-by-hour count at the cut-off bin edges
    # (exact only at bin edges; the cut-off is not one, so this is a bracket check)
    edges = np.arange(N_BINS + 1) * DI_STEP
    def above(h):
        return (h.sum() - np.concatenate([[0], np.cumsum(h)[:-1]])) / h.sum()     # share with value >= edge
    aD = above(hD["di"])
    j = int(cut_aoa // DI_STEP)
    if j < N_BINS and not (aD[j + 1] - 1e-6 <= rowD["outside_aoa"] <= aD[j] + 1e-6):
        raise SystemExit("hour-by-hour and histogram counts disagree")
    curve = {"di": edges}
    for g, (di, dk, _) in groups.items():
        curve[f"{g}_aoa"] = 1 - np.searchsorted(np.sort(di), edges, side="left") / len(di)
        curve[f"{g}_knn"] = 1 - np.searchsorted(np.sort(dk), edges, side="left") / len(dk)
    curve["D_australia_aoa"], curve["D_australia_knn"] = aD, above(hD["dk"])
    pd.DataFrame(curve).to_csv(out / "aoa_curve.csv", index=False, float_format="%.8g")

    names = ["outside_aoa", "outside_knn"] + [f"outside_{c}" for c in CATS]
    ymr = ymD[:, :n_y * 12].reshape(7, n_y, 12)
    by_y, by_m = ymr.sum(2), ymr.sum(1)
    def table(key, labels, arr):
        recs = []
        for i, lab in enumerate(labels):
            if arr[0, i] > 0:
                recs.append({key: lab, "weight": arr[0, i], **{nm: arr[1 + k, i] / arr[0, i] for k, nm in enumerate(names)},
                             "mean_di": arr[6, i] / arr[0, i]})
        return pd.DataFrame(recs)
    ty, tm, th = table("year", years, by_y), table("month", range(1, 13), by_m), table("hour", range(24), hrD)
    for df, f in ((ty, "aoa_by_year.csv"), (tm, "aoa_by_month.csv"), (th, "aoa_by_hour.csv")):
        df.to_csv(out / f, index=False, float_format="%.10g")
    trend = []
    for nm in names[:2] + ["outside_joint_only"]:
        yrs, s = ty["year"].values, ty[nm].values
        trend.append({"measure": nm, "years": f"{yrs[0]}-{yrs[-1]}", "first_decade": s[:10].mean(),
                      "last_decade": s[-10:].mean(),
                      "trend_pp_per_decade": np.polyfit(yrs, s, 1)[0] * 1000 if len(yrs) > 2 else np.nan})
    pd.DataFrame(trend).to_csv(out / "aoa_trend.csv", index=False, float_format="%.10g")

    pc = xr.Dataset(coords={"cell": cells.cell.values})
    for c_ in ("lat", "lon", "weight"):
        pc[c_] = ("cell", cells[c_].values)
    pc["valid"] = ("cell", cellD[0].astype(np.int32))
    pc["out_aoa"] = ("cell", cellD[1].astype(np.int32))
    pc["out_knn"] = ("cell", cellD[2].astype(np.int32))
    pc["out_joint_only"] = ("cell", cellD[3].astype(np.int32))
    pc["mean_di"] = ("cell", np.where(cellD[0] > 0, cellD[4] / np.maximum(cellD[0], 1), np.nan))
    pc.attrs["note"] = (f"hours per cell, {args.years[0]}-{args.years[1]}, variables {' '.join(vars_)}; "
                        "joint_only = outside AOA with every variable (and pair, if --pairs) inside")
    pc.to_netcdf(out / "aoa_per_cell.nc", encoding={v: {"zlib": True} for v in pc.data_vars})

    # ---------------- tower value (sample of D, cut-off and d_bar fixed)
    if args.value_every:
        Xs = np.concatenate([s[0] for s in samples]).astype("float64")
        ws = np.concatenate([s[1] for s in samples])
        oas = np.concatenate([s[2] for s in samples])
        base = (ws * oas).sum() / ws.sum()
        rows_v = []
        for s in np.unique(site):
            d1 = cKDTree(XA[site != s]).query(Xs, k=1, workers=args.workers)[0] / dbar
            sh = (ws * (d1 > cut_aoa)).sum() / ws.sum()
            rows_v.append({"site": s, "hours": int((site == s).sum()), "outside_with_all": base,
                           "outside_without": sh, "increase": sh - base})
        pd.DataFrame(rows_v).to_csv(out / "aoa_tower_value.csv", index=False, float_format="%.10g")
        print(f"tower value done on {len(Xs):,} sampled cell-hours (sample share {100 * base:.3f} %, "
              f"full {100 * rowD['outside_aoa']:.3f} %) ({time.time() - t_start:.0f} s)", flush=True)

    info = {"years": args.years, "variables": vars_, "tower_hours": len(XA), "d_bar": dbar, "aoa_cutoff": cut_aoa,
            "knn_cutoff": cut_knn, "k": K_NN, "knn_pct": KNN_PCT, "pairs": args.pairs is not None,
            "value_every": args.value_every, "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "aoa_run_info.json").write_text(json.dumps(info, indent=2))
    print(shares.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"done in {info['minutes']} min -> {out}")


if __name__ == "__main__":
    main()
