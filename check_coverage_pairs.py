"""
check_coverage_pairs.py
=======================

Self-check for coverage_pairs.py on a small test run (e.g. 2010-2011).
Rebuilds everything the slow, obvious way with its own code: the grid from
thresholds.csv (rain on 0.2 mm/h gauge steps), every envelope from the tower hours (pandas counts of bins, a
plain loop over count levels), and every ERA5 hour's status looked up one by one.

    python3 check_coverage_pairs.py --towers data --era5 /scratch/dt6/$USER/fc/era5_hourly \
        --single /scratch/dt6/$USER/fc/results_test --pairs /scratch/dt6/$USER/fc/results_pairs_test \
        --years 2010 2011

Checks, per pair:
  grid       pairs_grid.csv follows from thresholds.csv
  envelope   all 9 variants rebuilt independently; main one equals the stored one;
             each keeps >= 1 - trim of the tower hours
  shares     A, C and D (all 9 variants) and B (main) outside shares, to 1e-8 relative
  combo      combination-only for A, B, C, D (main), to 1e-8 relative
  per-cell   valid / outside / combination-only hours per cell, exactly
  by year    D outside and combination-only per local year, to 1e-8 relative
  LOTO       leave-one-site-out D share for the site with the biggest effect, to 1e-8
Prints PASS/FAIL per pair and ALL PASS at the end.
"""

import argparse
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARS = ["tair", "vpd", "qair", "psurf", "swdown", "lwdown", "precip", "wind"]
RANGE = "tower_p0.1-99.9"
BINS = {"bins20": 20, "bins40": 40, "bins80": 80}     # bins across the tower range
TRIMS = {"trim0": 0.0, "trim0.2": 0.002, "trim1": 0.01}
MAIN = ("bins40", "trim0.2")
FINE_PER_RANGE = 80
TOL = 1e-8


def axis(v, x):
    x = np.asarray(x, dtype="float64")
    if v == "precip":                       # gauge steps of 0.2 mm/h, then log10(r + 0.1)
        return np.log10(np.floor(np.maximum(x, 0) / 0.2 + 0.5) * 0.2 + 0.1)
    return x


def fine_bin(v, x, g):
    """Fine bin (80 per tower range) = floor((axis - start) / width), the grid's
    definition; -1 if beyond the grid or missing."""
    with np.errstate(invalid="ignore"):
        f = np.floor((axis(v, x) - g.start) / g.width)
    good = np.isfinite(f) & (f >= 0) & (f < g.n)
    return np.where(good, f, -1).astype(np.int16)


def coarse(f, per_range):
    """Fine bins -> bins of a grid with `per_range` bins per tower range (-1 stays -1)."""
    return np.where(f >= 0, f // (FINE_PER_RANGE // per_range), -1).astype(np.int64)


def keep_set(tower_bins, trim):
    """Envelope by the plain recipe: count tower hours per bin; drop count levels from
    the lowest up while the dropped bins hold at most trim of all hours."""
    counts = pd.Series(tower_bins).value_counts()
    total = counts.sum()
    cut = 0
    if trim > 0:
        for level in sorted(counts.unique()):
            if counts[counts <= level].sum() <= trim * total:
                cut = level
            else:
                break
    kept = set(counts.index[counts > cut])
    return kept, counts[counts > cut].sum() / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers", default="data")
    p.add_argument("--era5", required=True)
    p.add_argument("--single", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--years", nargs=2, type=int, required=True)
    args = p.parse_args()
    for f in ("coverage_pairs.py", "check_coverage_pairs.py"):
        print(f, hashlib.md5(Path(__file__).with_name(f).read_bytes()).hexdigest())
    print("numpy", np.__version__, "pandas", pd.__version__, "xarray", xr.__version__)
    single, res = Path(args.single), Path(args.pairs)

    # ---------------- grid
    thr = pd.read_csv(single / "thresholds.csv", float_precision="round_trip")
    grid = pd.read_csv(res / "pairs_grid.csv", float_precision="round_trip").set_index("variable")
    fails = 0
    for v in VARS:
        lo, hi = thr[(thr.range == RANGE) & (thr.quantity == v)][["lo", "hi"]].values[0]
        a, b = axis(v, [lo, hi])
        g = grid.loc[v]
        ok = (g.range_lo == lo and g.range_hi == hi and g.n == 240 and math.isclose(g.start, a - (b - a), rel_tol=1e-15, abs_tol=1e-12)
              and math.isclose(g.width, (b - a) / 80, rel_tol=1e-15))
        if not ok:
            print(f"FAIL grid {v}: {dict(g)} vs lo={lo} hi={hi}")
            fails += 1
    print(f"grid {'PASS' if fails == 0 else 'FAIL'}")
    rng = {v: tuple(grid.loc[v, ["range_lo", "range_hi"]]) for v in VARS}

    # ---------------- data
    sites = pd.read_csv(Path(args.towers) / "tower_sites.csv").set_index("site")
    cells = pd.read_csv(Path(args.era5) / "cells.csv")
    lat1, lon1 = np.radians(sites.lat.values)[:, None], np.radians(sites.lon.values)[:, None]
    lat2, lon2 = np.radians(cells.lat.values)[None], np.radians(cells.lon.values)[None]
    dist = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    sites["cell"] = cells.cell.values[dist.argmin(axis=1)]
    tw = pd.read_csv(Path(args.towers) / "tower_hourly.csv.gz", parse_dates=["time"])
    tw["cell"] = tw.site.map(sites.cell)
    ds = xr.concat([xr.open_dataset(Path(args.era5) / f"era5_hourly_{y}.nc")
                    for y in range(args.years[0], args.years[1] + 1)], "time")
    E = {v: ds[v].values.astype("float32") for v in VARS}            # (time, cell)
    times = pd.DatetimeIndex(ds.time.values)
    nt, nc = E["tair"].shape
    W = np.broadcast_to(cells.weight.values, (nt, nc))
    # ERA5 at each tower hour (B) and the tower cells (C), straight from the files
    t_pos = pd.Series(np.arange(nt), index=times)
    tw["pos"] = t_pos.reindex(tw.time).values
    inb = tw.pos.notna().values
    tcells = sorted(sites.cell.unique())
    # local year of every (time, cell)
    local_year = np.empty((nt, nc), np.int32)
    for off, g in cells.groupby("utc_offset"):
        local_year[:, g.index.values] = (times + pd.Timedelta(hours=off)).year.values[:, None]

    FA = {v: fine_bin(v, tw[v].values, grid.loc[v]) for v in VARS}      # tower hours
    FE = {v: fine_bin(v, E[v], grid.loc[v]) for v in VARS}               # ERA5 (time, cell)

    sh = pd.read_csv(res / "pairs_shares.csv")
    by = pd.read_csv(res / "pairs_by_year.csv")
    tv = pd.read_csv(res / "pairs_tower_value.csv")
    pc = xr.open_dataset(res / "pairs_per_cell.nc")
    hs = xr.open_dataset(res / "pairs_histograms.nc")

    def stored(name, var, group, col="outside"):
        r = sh[(sh.pair == name) & (sh.grid == var[0]) & (sh.trim == var[1]) & (sh.group == group)]
        return r[col].values[0]

    def rel(a, b):
        return abs(a - b) / max(abs(a), 1e-300) if a != 0 else abs(b)

    for x, y in itertools.combinations(VARS, 2):
        name = f"{x}-{y}"
        bad = []
        a_ok = tw[x].notna().values & tw[y].notna().values
        A = tw[a_ok]
        worst = 0.0
        for (gname, tname) in itertools.product(BINS, TRIMS):
            per = BINS[gname]
            m = 240 // (FINE_PER_RANGE // per)                 # coarse bins per axis
            code = lambda bx, by_: np.where((bx >= 0) & (by_ >= 0), bx * m + by_, -1)
            a_code = code(coarse(FA[x][a_ok], per), coarse(FA[y][a_ok], per))
            assert (a_code >= 0).all()
            kept, frac = keep_set(a_code, TRIMS[tname])
            if frac < 1 - TRIMS[tname] - 1e-12:
                bad.append(f"envelope {gname}/{tname} keeps {frac}")
            lut = np.zeros(m * m + 1, bool)                     # last entry: beyond the grid (code -1)
            lut[list(kept)] = True
            inside = lambda c: lut[c]
            if (gname, tname) == MAIN:
                env = hs[f"{name}_envelope"].values.astype(bool)
                mine = np.zeros((240, 240), bool)
                for c in kept:
                    mine[2 * (c // m):2 * (c // m) + 2, 2 * (c % m):2 * (c % m) + 2] = True
                if not (mine == env).all():
                    bad.append("stored main envelope differs")
            # A
            got = {"A_tower": 1 - inside(a_code).mean()}
            # C: tower cells, every hour
            cx, cy = E[x][:, tcells], E[y][:, tcells]
            okc = np.isfinite(cx) & np.isfinite(cy)
            c_code = code(coarse(FE[x][:, tcells][okc], per), coarse(FE[y][:, tcells][okc], per))
            got["C_tower_cells"] = 1 - inside(c_code).mean()
            # D: all of Australia, weighted, every hour
            okd = np.isfinite(E[x]) & np.isfinite(E[y])
            d_code = code(coarse(FE[x][okd], per), coarse(FE[y][okd], per))
            wd = W[okd]
            d_in = inside(d_code)
            got["D_australia"] = wd[~d_in].sum() / wd.sum()
            if (gname, tname) == MAIN:
                # B: ERA5 at the tower, same hours, both tower values measured
                bsel = a_ok & inb
                pos_b = tw.pos.values[bsel].astype(int)
                cel_b = tw.cell.values[bsel]
                bx, by_ = E[x][pos_b, cel_b], E[y][pos_b, cel_b]
                okb = np.isfinite(bx) & np.isfinite(by_)
                bx, by_ = bx[okb], by_[okb]
                b_in = inside(code(coarse(FE[x][pos_b, cel_b][okb], per), coarse(FE[y][pos_b, cel_b][okb], per)))
                got["B_matched"] = 1 - b_in.mean()
                # combination-only, every group
                inr = lambda v, val: (val >= rng[v][0]) & (val <= rng[v][1])
                a_in = inside(a_code)
                combos = {"A_tower": (~a_in & inr(x, A[x].values) & inr(y, A[y].values)).mean(),
                          "B_matched": (~b_in & inr(x, bx) & inr(y, by_)).mean(),
                          "C_tower_cells": (~inside(c_code) & inr(x, cx[okc]) & inr(y, cy[okc])).mean()}
                d_combo = ~d_in & inr(x, E[x][okd]) & inr(y, E[y][okd])
                combos["D_australia"] = wd[d_combo].sum() / wd.sum()
                for gr, val in combos.items():
                    e = rel(val, stored(name, MAIN, gr, "combination_only"))
                    worst = max(worst, e)
                    if e >= TOL:
                        bad.append(f"combo {gr} {val} vs {stored(name, MAIN, gr, 'combination_only')}")
                # per cell
                cell_idx = np.broadcast_to(np.arange(nc), (nt, nc))[okd]
                for k, mask in (("valid", np.ones(len(d_in), bool)), ("out", ~d_in), ("combo", d_combo)):
                    if not (np.bincount(cell_idx[mask], minlength=nc) == pc[f"{name}_{k}"].values).all():
                        bad.append(f"per-cell {k}")
                # by local year
                ly = local_year[okd]
                for r in by[by.pair == name].itertuples():
                    sel = ly == r.year
                    tot = wd[sel].sum()
                    for col_, mm in (("outside", ~d_in), ("combination_only", d_combo)):
                        e = rel(wd[sel & mm].sum() / tot, getattr(r, col_))
                        worst = max(worst, e)
                        if e >= TOL:
                            bad.append(f"by-year {r.year} {col_}")
                if set(by[by.pair == name].year) != set(range(args.years[0], args.years[1] + 1)):
                    bad.append("by-year rows")
                # leave one site out: the site with the biggest stored effect
                t = tv[tv.pair == name].sort_values("increase").iloc[-1]
                a_wo = a_code[A.site.values != t.site]
                kept_wo, _ = keep_set(a_wo, TRIMS[MAIN[1]])
                lut_wo = np.zeros(m * m + 1, bool)
                lut_wo[list(kept_wo)] = True
                in_wo = lut_wo[d_code]
                e = rel(wd[~in_wo].sum() / wd.sum(), t.outside_without)
                worst = max(worst, e)
                if e >= TOL:
                    bad.append(f"LOTO {t.site}")
            for gr, val in got.items():
                e = rel(val, stored(name, (gname, tname), gr))
                worst = max(worst, e)
                if e >= TOL:
                    bad.append(f"share {gname}/{tname} {gr}: recount {val!r} stored {stored(name, (gname, tname), gr)!r}")
        fails += bool(bad)
        print(f"{'PASS' if not bad else 'FAIL'} {name:15s} worst rel.err {worst:.1e}"
              + ("" if not bad else "\n      " + "\n      ".join(bad[:8])), flush=True)
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    main()
