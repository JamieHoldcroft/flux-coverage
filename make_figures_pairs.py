"""
make_figures_pairs.py
=====================

Step 5 figures (laptop or Gadi). Turns the coverage_pairs.py results folder into
figures and a short summary. Uses the helpers and colours of make_figures.py.

    python make_figures_pairs.py --results results_pairs --towers data --out results_pairs/figures

Figures (in --out):
    figp01_matrix.png         % of Australian land-hours outside the tower envelope, and combination-only, every pair
    figp02_envelopes.png      the 8 pairs with the largest share outside: Australia's density and the tower envelope
    figp03_maps.png           per cell: % of hours outside the envelope, same 8 pairs
    figp04_maps_combo.png     per cell: % combination-only, the 8 pairs with the most of it
    figp05_decomposition.png  % outside: matched -> tower cells -> Australia, every pair
    figp06_sensitivity.png    % outside under the 9 envelope variants, every pair
    figp07_trend.png          % outside by year, the 8 top pairs
    figp08_tower_value.png    increase in % outside when each site is left out (sites x pairs)
    summary_pairs.md          the key numbers as tables
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr

from make_figures import (VARIABLES, LABEL, SHORT, C_TOWER, C_AUS, C_CELLS, C_MATCH, INK, INK2, SEQ,
                          panels, _map, md_table)

MAIN_GRID, MAIN_TRIM = "bins40", "trim0.2"
RAIN_TICKS = [0, 0.2, 1, 10, 100]
N_TOP = 8


def pname(x, y):
    return f"{SHORT[x]} × {SHORT[y]}"


def fig_matrix(D, out):
    n = len(VARIABLES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    vmax = max(1.0, 100 * D.outside.max())
    for ax, col, title in zip(axes, ("outside", "combination_only"),
                              ("% outside the 2-D tower envelope", "% combination-only (both variables inside their own range)")):
        m = np.full((n, n), np.nan)
        for r in D.itertuples():
            i, j = VARIABLES.index(r.x), VARIABLES.index(r.y)
            m[j, i] = 100 * getattr(r, col)
        im = ax.imshow(m, cmap=SEQ, vmin=0, vmax=vmax)
        for i, j in itertools.product(range(n), range(n)):
            if np.isfinite(m[j, i]):
                ax.text(i, j, f"{m[j, i]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[j, i] > 0.6 * vmax else INK)
        ax.set_xticks(range(n), [SHORT[v] for v in VARIABLES], rotation=45)
        ax.set_yticks(range(n), [SHORT[v] for v in VARIABLES])
        ax.grid(False)
        ax.set_title(title, loc="left", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="% of Australian land-hours")
    fig.savefig(out / "figp01_matrix.png")
    plt.close(fig)


def _axis_ticks(ax, v, which):
    if v != "precip":
        return
    pos = np.log10(np.array(RAIN_TICKS) + 0.1)
    (ax.set_xticks if which == "x" else ax.set_yticks)(pos, [f"{t:g}" for t in RAIN_TICKS])


def fig_envelopes(hs, grid, top, out):
    fig, axes = panels(len(top), 4, w=3.2, h=2.9)
    for ax, (x, y) in zip(axes, top):
        p = f"{x}-{y}"
        d = hs[f"{p}_D"].values
        a = hs[f"{p}_A"].values
        env = hs[f"{p}_envelope"].values
        bx, by = hs[f"b_{x}"].values, hs[f"b_{y}"].values
        wx, wy = grid.loc[x, "width"], grid.loc[y, "width"]
        ex = np.append(bx - wx / 2, bx[-1] + wx / 2)
        ey = np.append(by - wy / 2, by[-1] + wy / 2)
        dens = np.where(d > 0, d / d.sum(), np.nan)
        ax.pcolormesh(ex, ey, dens.T, cmap="Greys", norm=mcolors.LogNorm(vmin=1e-7, vmax=np.nanmax(dens)),
                      shading="flat", rasterized=True)
        # outside-the-envelope Australian hours in orange
        outside = np.where((d > 0) & (env == 0), dens, np.nan)
        ax.pcolormesh(ex, ey, outside.T, cmap=mcolors.ListedColormap([C_AUS]), shading="flat", rasterized=True)
        ax.contour(bx, by, env.T.astype(float), levels=[0.5], colors=C_TOWER, linewidths=1.0)
        lo_x, hi_x = grid.loc[x, ["axis_lo", "axis_hi"]]
        lo_y, hi_y = grid.loc[y, ["axis_lo", "axis_hi"]]
        ax.plot([lo_x, hi_x, hi_x, lo_x, lo_x], [lo_y, lo_y, hi_y, hi_y, lo_y], color=INK2, lw=0.7, ls="--")
        used = (d > 0) | (a > 0)
        ix, iy = np.nonzero(used.any(axis=1))[0], np.nonzero(used.any(axis=0))[0]
        ax.set_xlim(ex[ix[0]], ex[ix[-1] + 1])
        ax.set_ylim(ey[iy[0]], ey[iy[-1] + 1])
        _axis_ticks(ax, x, "x")
        _axis_ticks(ax, y, "y")
        ax.set_xlabel(LABEL[x])
        ax.set_ylabel(LABEL[y])
        ax.grid(False)
        ax.set_title(pname(x, y), loc="left", fontsize=9)
    fig.suptitle("Grey: Australia (ERA5, log density). Blue line: tower envelope. Orange: Australian hours outside it. "
                 "Dashed: the two 1-variable tower ranges.", fontsize=8, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "figp02_envelopes.png")
    plt.close(fig)


def fig_maps(pc, sites, pairs, key, out, name):
    lon, lat = pc.lon.values, pc.lat.values
    fig, axes = panels(len(pairs), 4, w=3.0, h=2.6)
    for ax, (x, y) in zip(axes, pairs):
        p = f"{x}-{y}"
        valid = pc[f"{p}_valid"].values
        share = np.where(valid > 0, 100 * pc[f"{p}_{key}"].values / np.maximum(valid, 1), np.nan)
        vmax = max(1.0, np.nanpercentile(share, 98))
        sc = _map(ax, lon, lat, share, sites, vmax, pname(x, y))
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"% of hours {'outside the tower envelope' if key == 'out' else 'combination-only'} per cell "
                 "(dots: towers)", fontsize=9, x=0.01, ha="left")
    fig.savefig(out / name)
    plt.close(fig)


def fig_dots(df, cols, colours, labels, xlabel, out, name, title):
    """One row per pair, dots per column (shares in %)."""
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(df) + 1.2))
    yy = np.arange(len(df))
    for c, colr, lab in zip(cols, colours, labels):
        ax.scatter(100 * df[c].values, yy, s=22, color=colr, label=lab, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(yy, [pname(x, y) for x, y in zip(df.x, df.y)])
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(out / name)
    plt.close(fig)


def fig_trend(by_year, trend, top, out):
    fig, axes = panels(len(top), 4, w=3.0, h=2.2)
    for ax, (x, y) in zip(axes, top):
        p = f"{x}-{y}"
        b = by_year[by_year.pair == p]
        t = trend[trend.pair == p].iloc[0]
        ax.plot(b.year, 100 * b.outside, color=C_AUS, lw=1.5)
        ax.plot(b.year, 100 * b.combination_only, color=INK2, lw=1.0, ls=":")
        if len(b) > 2:
            k = np.polyfit(b.year, 100 * b.outside, 1)
            ax.plot(b.year, np.polyval(k, b.year), color=INK, lw=0.8, ls="--")
        ax.set_title(f"{pname(x, y)}  ({t.trend_outside_pp_per_decade:+.2f} pp/decade)", loc="left", fontsize=8)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("% of land-hours")
    fig.suptitle("Outside the envelope (orange; dashed = linear trend) and combination-only (dotted), by year",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "figp07_trend.png")
    plt.close(fig)


def fig_tower_value(tv, out):
    m = tv.pivot(index="site", columns="pair", values="increase") * 100
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]
    m = m[[f"{x}-{y}" for x, y in itertools.combinations(VARIABLES, 2) if f"{x}-{y}" in m.columns]]
    fig, ax = plt.subplots(figsize=(10, 0.2 * len(m) + 1.8))
    vmax = max(0.01, np.nanpercentile(m.values, 99.5))
    im = ax.imshow(m.values, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(m)), m.index, fontsize=7)
    ax.set_xticks(range(m.shape[1]), [pname(*c.split("-")) for c in m.columns], rotation=90, fontsize=7)
    ax.grid(False)
    ax.set_title("Increase in % of Australian land-hours outside the envelope when the site is left out "
                 "(sites sorted by total)", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6, label="percentage points")
    fig.savefig(out / "figp08_tower_value.png")
    plt.close(fig)


def write_summary(sh, trend, tv, info, out):
    main = sh[(sh.grid == MAIN_GRID) & (sh.trim == MAIN_TRIM)]
    t = main.pivot(index="pair", columns="group", values="outside") * 100
    t["combination_only"] = main[main.group == "D_australia"].set_index("pair").combination_only * 100
    D = sh[sh.group == "D_australia"]
    rng = D.groupby("pair").outside.agg(["min", "max"]) * 100
    t["variants_min"], t["variants_max"] = rng["min"], rng["max"]
    t["trend_pp_per_decade"] = trend.set_index("pair").trend_outside_pp_per_decade
    best = tv.sort_values("increase").groupby("pair").tail(1).set_index("pair")
    t["most_valuable_site"] = best.site + " (+" + (100 * best.increase).map("{:.3f}".format) + ")"
    t = t.sort_values("D_australia", ascending=False)
    t.insert(0, "pair", [pname(*p.split("-")) for p in t.index])
    t = t[["pair", "D_australia", "combination_only", "B_matched", "C_tower_cells", "variants_min", "variants_max",
           "trend_pp_per_decade", "most_valuable_site"]]
    lines = [f"# Pair coverage summary ({info['years'][0]}-{info['years'][1]})", "",
             "Share of hours (%) outside the 2-D tower envelope (main: 40 bins across each tower range, "
             "sparsest bins holding 0.2 % of tower hours dropped). D = all Australia (area weighted), "
             "B = ERA5 at the towers in the measured hours, C = ERA5 at the tower cells, all hours. "
             "Combination-only = outside the envelope but inside both 1-variable ranges. "
             "Variants = min and max over 20/40/80 bins x 0/0.2/1 % trimmed. "
             "Most valuable site = biggest increase in D (percentage points) when left out.", "",
             md_table(t.reset_index(drop=True), "{:.3f}"), ""]
    (out / "summary_pairs.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--towers", default="data")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import json
    info = json.loads((res / "pairs_run_info.json").read_text())
    sh = pd.read_csv(res / "pairs_shares.csv")
    grid = pd.read_csv(res / "pairs_grid.csv").set_index("variable")
    by_year = pd.read_csv(res / "pairs_by_year.csv")
    trend = pd.read_csv(res / "pairs_trend.csv")
    tv = pd.read_csv(res / "pairs_tower_value.csv")
    pc = xr.open_dataset(res / "pairs_per_cell.nc")
    hs = xr.open_dataset(res / "pairs_histograms.nc")
    sites = pd.read_csv(Path(args.towers) / "tower_sites.csv")

    main_ = sh[(sh.grid == MAIN_GRID) & (sh.trim == MAIN_TRIM)]
    D = main_[main_.group == "D_australia"].sort_values("outside", ascending=False)
    top = list(zip(D.x, D.y))[:N_TOP]
    top_combo = list(zip(*D.sort_values("combination_only", ascending=False)[["x", "y"]].values.T))[:N_TOP]

    fig_matrix(D, out)
    fig_envelopes(hs, grid, top, out)
    fig_maps(pc, sites, top, "out", out, "figp03_maps.png")
    fig_maps(pc, sites, top_combo, "combo", out, "figp04_maps_combo.png")
    wide = main_.pivot(index="pair", columns="group", values="outside")
    wide = wide.join(D.set_index("pair")[["x", "y"]]).loc[D.pair]
    fig_dots(wide, ["B_matched", "C_tower_cells", "D_australia"], [C_MATCH, C_CELLS, C_AUS],
             ["ERA5 at towers, same hours", "ERA5 at tower cells, all hours", "All Australia"],
             "% of hours outside the tower envelope", out, "figp05_decomposition.png",
             "Point vs grid (matched), which years (tower cells), which places (Australia)")
    var = sh[sh.group == "D_australia"].copy()
    var["variant"] = var.grid + "/" + var.trim
    wv = var.pivot(index="pair", columns="variant", values="outside").join(D.set_index("pair")[["x", "y"]]).loc[D.pair]
    cols = [f"{g}/{t}" for g in ("bins20", "bins40", "bins80") for t in ("trim0", "trim0.2", "trim1")]
    shades = {"bins20": "#9ec5f0", "bins40": C_TOWER, "bins80": "#123f73"}
    fig_dots(wv, cols, [shades[c.split("/")[0]] for c in cols], cols,
             "% of Australian land-hours outside the envelope", out, "figp06_sensitivity.png",
             "Envelope variants (bins across the tower range / share of tower hours trimmed)")
    fig_trend(by_year, trend, top, out)
    fig_tower_value(tv, out)
    write_summary(sh, trend, tv, info, out)
    print(f"figures and summary_pairs.md -> {out}")


if __name__ == "__main__":
    main()
