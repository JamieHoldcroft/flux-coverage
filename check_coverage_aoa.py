"""
check_coverage_aoa.py
=====================

Independent check of coverage_aoa.py, for a TEST run made with --save-di
(e.g. 2 years). Recomputes, with separate code and without kd-trees:

  1. the variable transform and standardisation (aoa_settings.csv)
  2. d_bar (fresh random pairs, different seed: must agree within 0.5 %)
  3. the DI and 10th-neighbour distance of random tower hours (own site / own hour
     left out) and of random ERA5 hours, by brute force over every tower hour
  4. the AOA cut-off (box-plot whisker, written here from the fivenum definition
     as medians of the lower and upper halves) and the kNN cut-off
  5. groups A-D: share outside AOA / kNN, mean DI, and the breakdown
     (single / pair / joint_only), recounted from the saved DIs with this script's
     own range and envelope tests, own local-time and area-weight bookkeeping
  6. D by local year, month and hour, and per cell

    python3 check_coverage_aoa.py --towers data --era5 ERA5_DIR --single SINGLE_DIR \
        [--pairs PAIRS_DIR] --run RESULTS_AOA [--n-brute 2000]

Prints PASS/FAIL per check and ALL PASS at the end.
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ORDER = ["tair", "vpd", "qair", "psurf", "swdown", "lwdown", "precip", "wind"]
FAILS = []


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def close(a, b, rtol=1e-9, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def axis(v, x):
    """Rain: nearest 0.2 mm/h step, then log10(+0.1). Written independently of coverage_pairs."""
    x = np.asarray(x, dtype="float64")
    if v == "precip":
        # number of 0.2 mm/h steps. NB x / 0.2 (not x * 5) on purpose: exact half steps (0.3, 0.7 mm/h ...)
        # then round as coverage_pairs.py does (0.3 -> 0.2, 0.5 -> 0.6: floating point). Same rule for towers
        # and ERA5, so no bias; kept so the AOA run uses the same rain axis as the finished pairs run.
        steps = np.floor(np.maximum(x, 0.0) / 0.2 + 0.5)
        return np.log10(steps * 0.2 + 0.1)
    return x


def brute(Q, X, k=10, block=24):
    """Exact sorted distances from each row of Q to its k nearest rows of X (numpy only)."""
    out = np.empty((len(Q), k))
    xx = (X ** 2).sum(1)
    for s in range(0, len(Q), block):
        q = Q[s:s + block]
        d2 = (q ** 2).sum(1)[:, None] + xx[None] - 2 * q @ X.T
        # candidates from the fast expansion, then exact distances (the expansion loses precision near 0)
        idx = np.argpartition(d2, k + 5, axis=1)[:, :k + 6]
        exact = np.sqrt(((q[:, None, :] - X[idx]) ** 2).sum(2))
        out[s:s + block] = np.sort(exact, axis=1)[:, :k]
    return out


def whisker(x, coef=1.5):
    """Box-plot upper whisker with Tukey hinges = medians of the lower / upper halves
    (each half includes the median when n is odd), as R's fivenum."""
    x = np.sort(x)
    n = len(x)
    half = (n + 1) // 2
    lo_h, up_h = np.median(x[:half]), np.median(x[n - half:])
    return x[x <= up_h + coef * (up_h - lo_h)].max(), lo_h, up_h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--towers", default="data")
    p.add_argument("--era5", required=True)
    p.add_argument("--single", required=True)
    p.add_argument("--pairs", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--n-brute", type=int, default=2000)
    a = p.parse_args()
    run = Path(a.run)
    print("versions: numpy", np.__version__, "pandas", pd.__version__, "xarray", xr.__version__)
    for f in sorted(Path(__file__).parent.glob("coverage_aoa.py")) + [Path(__file__)]:
        print("md5", f.name, hashlib.md5(f.read_bytes()).hexdigest())
    info = json.loads((run / "aoa_run_info.json").read_text())
    st = pd.read_csv(run / "aoa_settings.csv", float_precision="round_trip")
    th = pd.read_csv(run / "aoa_threshold.csv", float_precision="round_trip").iloc[0]
    vars_ = list(st.variable)
    years = list(range(info["years"][0], info["years"][1] + 1))
    rng = np.random.default_rng(7)

    # ---------------- 1. towers and standardisation
    tw = pd.read_csv(Path(a.towers) / "tower_hourly.csv.gz")
    sites = pd.read_csv(Path(a.towers) / "tower_sites.csv")
    tw = tw[tw.site.isin(sites.site)]
    keep = np.ones(len(tw), bool)
    for v in vars_:
        keep &= tw[v].notna().values
    tw = tw[keep].reset_index(drop=True)
    T = np.column_stack([axis(v, tw[v].values) for v in vars_])
    mu, sd = T.mean(0), T.std(0, ddof=1)
    report("tower hours", len(tw) == th.tower_hours, f"{len(tw):,}")
    report("tower mean / sd", close(mu, st.tower_mean, 1e-10) and close(sd, st.tower_sd, 1e-10))
    X = (T - mu) / sd
    site = tw.site.values

    # ---------------- 2. d_bar
    i = rng.integers(0, len(X), 2_000_000)
    j = rng.integers(0, len(X), 2_000_000)
    ok = i != j
    dbar = np.sqrt(((X[i[ok]] - X[j[ok]]) ** 2).sum(1)).mean()
    report("d_bar", abs(dbar / th.d_bar - 1) < 0.005, f"check {dbar:.4f} vs run {th.d_bar:.4f}")
    dbar = th.d_bar                                     # use the run's value from here on (it only scales DI)

    # ---------------- 3+4. tower DIs, cut-offs
    G = np.load(run / "di_groups.npz")
    di_loto, di_all, dk_all = G["A_tower_loto_di"], G["A_tower_all_di"], G["A_tower_all_dk"]
    pick = rng.choice(len(X), a.n_brute, replace=False)
    loto_b = np.empty(len(pick))
    for s in np.unique(site[pick]):
        sel = pick[site[pick] == s]
        loto_b[site[pick] == s] = brute(X[sel], X[site != s], k=1)[:, 0] / dbar
    allb = brute(X[pick], X, k=11)                       # includes the hour itself at 0
    report("tower DI, own site left out (brute force)", close(loto_b, di_loto[pick], 1e-7, 1e-9),
           f"max diff {np.abs(loto_b - di_loto[pick]).max():.2e}")
    report("tower DI, own hour left out (brute force)", close(allb[:, 1] / dbar, di_all[pick], 1e-7, 1e-9))
    report("tower 10th-neighbour distance (brute force)", close(allb[:, 10] / dbar, dk_all[pick], 1e-7, 1e-9))
    w, lo_h, up_h = whisker(di_loto)
    report("AOA cut-off = box-plot upper whisker of LOTO DI", w == th.aoa_cutoff and close([lo_h, up_h],
           [th.loto_lower_hinge, th.loto_upper_hinge]), f"{w:.6f} vs {th.aoa_cutoff:.6f}")
    srt, pos = np.sort(dk_all), (len(dk_all) - 1) * 0.998          # linear interpolation between these two
    lo_v, hi_v = srt[int(np.floor(pos))], srt[int(np.ceil(pos))]
    report("kNN cut-off = 99.8th percentile", lo_v <= th.knn_cutoff <= hi_v and
           close(lo_v + (pos - np.floor(pos)) * (hi_v - lo_v), th.knn_cutoff, 1e-12), f"{th.knn_cutoff:.6f}")
    cut_a, cut_k = th.aoa_cutoff, th.knn_cutoff

    # ---------------- ranges and envelopes (own reading)
    thr = pd.read_csv(Path(a.single) / "thresholds.csv", float_precision="round_trip")
    rng_ = {v: thr[(thr.range == "tower_p0.1-99.9") & (thr.quantity == v)][["lo", "hi"]].values[0] for v in vars_}
    env = {}
    if a.pairs:
        g = pd.read_csv(Path(a.pairs) / "pairs_grid.csv", float_precision="round_trip").set_index("variable")
        with xr.open_dataset(Path(a.pairs) / "pairs_histograms.nc") as hs:
            for x, y in itertools.combinations([v for v in ORDER if v in vars_], 2):
                env[(x, y)] = hs[f"{x}-{y}_envelope"].values.astype(bool)
        grid = {v: (g.loc[v, "start"], g.loc[v, "width"], int(g.loc[v, "n"])) for v in vars_}

    def categories(vals):
        n = len(vals[vars_[0]])
        single = np.zeros(n, bool)
        for v in vars_:
            single |= ~((vals[v] >= rng_[v][0]) & (vals[v] <= rng_[v][1]))
        pair = np.zeros(n, bool)
        if env:
            b = {}
            for v in vars_:
                st_, wd, nb = grid[v]
                f = np.floor((axis(v, vals[v]) - st_) / wd)
                b[v] = np.where((f >= 0) & (f < nb), f, -1).astype(int)
            for (x, y), m in env.items():
                inside = (b[x] >= 0) & (b[y] >= 0)
                ok = np.zeros(n, bool)
                ok[inside] = m[b[x][inside], b[y][inside]]
                pair |= ~ok
        return np.where(single, 0, np.where(pair, 1, 2))

    shares = pd.read_csv(run / "aoa_shares.csv").set_index("group")

    def check_group(name, di, dk, cat, wt=None):
        wt = np.ones(len(di)) if wt is None else wt
        s = shares.loc[name]
        oa = di > cut_a
        got = [(wt * oa).sum() / wt.sum(), (wt * (dk > cut_k)).sum() / wt.sum(), (wt * di).sum() / wt.sum()]
        exp = [s.outside_aoa, s.outside_knn, s.mean_di]
        cats = [(wt * (oa & (cat == c))).sum() / wt.sum() for c in range(3)]
        if not env:
            cats = [cats[0], np.nan, cats[1] + cats[2]]
        exp_c = [s.outside_single, s.outside_pair, s.outside_joint_only]
        report(f"{name}: shares, mean DI, breakdown", close(got, exp, 1e-8) and close(cats, exp_c, 1e-8)
               and int(s.n) == len(di), f"AOA {100 * got[0]:.3f} %, kNN {100 * got[1]:.3f} %")

    A_vals = {v: tw[v].values.astype("float64") for v in vars_}
    A_cat = categories(A_vals)
    check_group("A_tower_loto", di_loto, dk_all, A_cat)
    check_group("A_tower_all", di_all, dk_all, A_cat)

    # ---------------- B and C straight from the ERA5 files (not the cache)
    cells = pd.read_csv(Path(a.era5) / "cells.csv")
    lat, lon = np.radians(sites.lat.values), np.radians(sites.lon.values)
    clat, clon = np.radians(cells.lat.values), np.radians(cells.lon.values)
    hv = np.sin((clat[None] - lat[:, None]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(clat[None]) * \
        np.sin((clon[None] - lon[:, None]) / 2) ** 2
    site_cell = dict(zip(sites.site, hv.argmin(1)))
    tcells = np.unique(list(site_cell.values()))
    parts = []
    for y in years:
        with xr.open_dataset(Path(a.era5) / f"era5_hourly_{y}.nc") as ds:
            parts.append(ds[vars_].isel(cell=tcells).load())
    E = xr.concat(parts, "time")
    etimes = pd.DatetimeIndex(E.time.values)
    C_vals = {v: E[v].values.astype("float32").ravel() for v in vars_}
    okC = np.ones(len(C_vals[vars_[0]]), bool)
    for v in vars_:
        okC &= np.isfinite(C_vals[v])
    C_vals = {v: C_vals[v][okC] for v in vars_}
    tpos = etimes.get_indexer(pd.to_datetime(tw.time))
    ccol = np.searchsorted(tcells, tw.site.map(site_cell).values)
    hit = tpos >= 0
    B_vals = {v: np.full(len(tw), np.nan, "float32") for v in vars_}
    for v in vars_:
        B_vals[v][hit] = E[v].values[tpos[hit], ccol[hit]]
    okB = np.ones(len(tw), bool)
    for v in vars_:
        okB &= np.isfinite(B_vals[v])
    B_vals = {v: B_vals[v][okB] for v in vars_}
    for name, vals in (("B_matched", B_vals), ("C_tower_cells", C_vals)):
        di, dk = G[f"{name}_di"], G[f"{name}_dk"]
        if len(di) != len(vals[vars_[0]]):
            report(f"{name}: hour count", False, f"{len(di)} vs {len(vals[vars_[0]])}")
            continue
        Q = np.column_stack([(axis(v, vals[v]) - mu[k]) / sd[k] for k, v in enumerate(vars_)])
        pk = rng.choice(len(Q), min(a.n_brute // 2, len(Q)), replace=False)
        bd = brute(Q[pk], X, k=10) / dbar
        report(f"{name}: DI and kNN distance (brute force)", close(bd[:, 0], di[pk], 1e-7, 1e-9)
               and close(bd[:, 9], dk[pk], 1e-7, 1e-9))
        check_group(name, di, dk, categories(vals))

    # ---------------- D: every cell-hour, from the saved DIs
    wcell = np.cos(np.radians(cells.lat.values))
    report("cell weights = cos(lat)", close(wcell, cells.weight.values, 1e-12))
    off = cells.utc_offset.values
    allD = {k: [] for k in ("di", "dk", "w", "cat", "ly", "lm", "lh", "cell")}
    for y in years:
        sv = np.load(run / f"di_{y}.npy")
        with xr.open_dataset(Path(a.era5) / f"era5_hourly_{y}.nc") as ds:
            t = pd.DatetimeIndex(ds.time.values)
            vals = {v: ds[v].values.astype("float32").ravel() for v in vars_}
        nt, nc = len(t), len(cells)
        okD = np.ones(nt * nc, bool)
        for v in vars_:
            okD &= np.isfinite(vals[v])
        report(f"D {y}: saved DI exactly where all variables are valid", (np.isfinite(sv[0]) == okD).all())
        # brute force on random hours of this year
        pk = rng.choice(np.nonzero(okD)[0], a.n_brute // 2, replace=False)
        Q = np.column_stack([(axis(v, vals[v][pk]) - mu[k]) / sd[k] for k, v in enumerate(vars_)])
        bd = brute(Q, X, k=10) / dbar
        report(f"D {y}: DI and kNN distance (brute force)", close(bd[:, 0], sv[0, pk], 1e-7, 1e-9)
               and close(bd[:, 9], sv[1, pk], 1e-7, 1e-9), f"max diff {np.abs(bd[:, 0] - sv[0, pk]).max():.2e}")
        tt = np.repeat(t.values, nc)[okD]
        cc = np.tile(np.arange(nc), nt)[okD]
        local = pd.DatetimeIndex(tt + (off[cc] * 3600).astype("timedelta64[s]"))
        for k, val in (("di", sv[0, okD]), ("dk", sv[1, okD]), ("w", wcell[cc]),
                       ("cat", categories({v: vals[v][okD] for v in vars_})),
                       ("ly", local.year.values), ("lm", local.month.values), ("lh", local.hour.values), ("cell", cc)):
            allD[k].append(val)
    D = {k: np.concatenate(v) for k, v in allD.items()}
    check_group("D_australia", D["di"], D["dk"], D["cat"], D["w"])

    oa, ok_ = D["di"] > cut_a, D["dk"] > cut_k
    df = pd.DataFrame({"year": D["ly"], "month": D["lm"], "hour": D["lh"], "w": D["w"],
                       "a": D["w"] * oa, "k": D["w"] * ok_, "j": D["w"] * (oa & (D["cat"] == 2)), "d": D["w"] * D["di"]})
    for key, f in (("year", "aoa_by_year.csv"), ("month", "aoa_by_month.csv"), ("hour", "aoa_by_hour.csv")):
        # by year and by month: the few local hours after the last year are left out (as in the run)
        g = (df if key == "hour" else df[df.year.isin(years)]).groupby(key)[["w", "a", "k", "j", "d"]].sum()
        r = pd.read_csv(run / f).set_index(key)
        jcol = "outside_joint_only"
        exp_j = r[jcol] + (0 if env else r["outside_pair"].fillna(0))
        report(f"D by {key}", list(g.index) == list(r.index) and close(g.a / g.w, r.outside_aoa, 1e-8)
               and close(g.k / g.w, r.outside_knn, 1e-8) and close(g.j / g.w, exp_j, 1e-8)
               and close(g.d / g.w, r.mean_di, 1e-8))
    pc = xr.open_dataset(run / "aoa_per_cell.nc")
    nc = len(cells)
    report("D per cell", (np.bincount(D["cell"], minlength=nc) == pc.valid.values).all()
           and (np.bincount(D["cell"], weights=oa, minlength=nc) == pc.out_aoa.values).all()
           and (np.bincount(D["cell"], weights=ok_, minlength=nc) == pc.out_knn.values).all()
           and (np.bincount(D["cell"], weights=oa & (D["cat"] == 2), minlength=nc) == pc.out_joint_only.values).all())

    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
