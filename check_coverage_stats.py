"""
check_coverage_stats.py
=======================

Self-check for coverage_stats.py on a small test run (e.g. 2010-2011):
recomputes every Australia share, the per-cell counts and the D quantiles the
slow, obvious way (straight from the ERA5 files) and compares.

    python3 check_coverage_stats.py --era5 /scratch/dt6/$USER/fc/era5_hourly \
        --results /scratch/dt6/$USER/fc/results_test --years 2010 2011

Prints PASS/FAIL per check. Shares must agree to 1e-8 relative (exact, stored with 10 digits);
quantiles to within one fine bin.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import coverage_stats as cs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--years", nargs=2, type=int, required=True)
    args = p.parse_args()
    res = Path(args.results)
    cells = pd.read_csv(Path(args.era5) / "cells.csv")
    thr = pd.read_csv(res / "thresholds.csv")
    sh = pd.read_csv(res / "shares.csv")
    rg = pd.read_csv(res / "ranges.csv")
    tv = pd.read_csv(res / "tower_value.csv")
    pc = xr.open_dataset(res / "per_cell.nc")
    ds = xr.concat([xr.open_dataset(Path(args.era5) / f"era5_hourly_{y}.nc")
                    for y in range(args.years[0], args.years[1] + 1)], "time")
    t = pd.DatetimeIndex(ds.time.values)
    w = cells.weight.values
    site = tv.site.iloc[0]
    fails = 0
    for q in cs.QUANTITIES:
        x = ds[q].values.astype("float32")
        ok = np.isfinite(x)
        W = np.broadcast_to(w, x.shape)
        worst = 0.0
        for rname in thr.range.unique():
            if rname.startswith("without_") and rname != f"without_{site}":
                continue
            lo, hi = thr[(thr.range == rname) & (thr.quantity == q)][["lo", "hi"]].values[0]
            direct = (W[x < lo].sum() + W[x > hi].sum()) / W[ok].sum()
            if rname.startswith("without_"):
                got = tv[(tv.site == site) & (tv.quantity == q)].outside_without.values[0]
            else:
                got = sh[(sh.range == rname) & (sh.quantity == q) & (sh.group == "D_australia")].outside.values[0]
            worst = max(worst, abs(direct - got) / max(direct, 1e-12) if direct > 0 else abs(got))
        lo, hi = thr[(thr.range == cs.MAIN) & (thr.quantity == q)][["lo", "hi"]].values[0]
        cell_ok = bool(((pc[f"{q}_below"].sum("year") + pc[f"{q}_above"].sum("year")).values
                        == ((x < lo) | (x > hi)).sum(0)).all())
        r = rg[(rg.group == "D_australia") & (rg.quantity == q)].iloc[0]
        xv, wv = x[ok], W[ok]
        o = np.argsort(xv)
        cw = np.cumsum(wv[o]) / wv.sum()
        e = cs.FINE_EDGES[q]
        qerr = []
        for pp in (0.1, 50, 99.9):
            exact = xv[o][min(np.searchsorted(cw, pp / 100), len(o) - 1)]
            width = np.diff(e)[min(max(np.searchsorted(e, exact) - 1, 0), len(e) - 2)]
            qerr.append(abs(exact - r[f"p{pp:g}"]) / width)
        mean_err = abs(np.average(xv, weights=wv) - r["mean"])
        good = worst < 1e-8 and cell_ok and max(qerr) <= 1.0 and mean_err < 1e-6 * max(1, abs(r["mean"]))
        fails += not good
        print(f"{'PASS' if good else 'FAIL'} {q:11s} shares rel.err {worst:.1e}  per-cell {cell_ok}  "
              f"quantiles within {max(qerr):.2f} bins  mean err {mean_err:.1e}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")


if __name__ == "__main__":
    main()
