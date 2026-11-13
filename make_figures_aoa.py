"""
make_figures_aoa.py
===================

Step 6 figures (laptop or Gadi). Turns a coverage_aoa.py results folder into
figures and a short summary, in the style and colours of make_figures.py.

    python make_figures_aoa.py --results results_aoa --towers data --out results_aoa/figures

Figures (in --out):
    figa01_maps.png           per cell: % of hours outside AOA, outside kNN-10, joint-only; mean DI
    figa02_curve.png          % of hours above each DI value (towers, ERA5 at towers, Australia), cut-offs marked
    figa03_decomposition.png  % outside: towers -> matched -> tower cells -> Australia; breakdown of the uncovered hours
    figa04_by_month_hour.png  % outside by local month and local hour
    figa05_trend.png          % outside by year
    figa06_tower_sites.png    per site: % of its own hours outside when it is left out (how unlike the rest it is)
    figa07_tower_value.png    per site: extra % of Australia uncovered without it
    summary_aoa.md            the key numbers as tables
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from make_figures import SHORT, C_TOWER, C_AUS, C_CELLS, C_MATCH, INK, INK2, panels, _map, md_table

GROUPS = [("A_tower_loto", "towers (own site left out)", C_TOWER, "-"),
          ("A_tower_all", "towers (own hour left out)", C_TOWER, ":"),
          ("B_matched", "ERA5 at the towers, same hours", C_MATCH, "-"),
          ("C_tower_cells", "ERA5 at the tower cells, all hours", C_CELLS, "-"),
          ("D_australia", "ERA5, all Australia", C_AUS, "-")]
CAT_COL = {"outside_single": "#f3b37a", "outside_pair": "#e07b39", "outside_joint_only": "#8c3b0f"}
CAT_LAB = {"outside_single": "a variable outside its own range", "outside_pair": "all inside, a pair outside its envelope",
           "outside_joint_only": "every variable and pair inside (joint only)"}


def fig_maps(pc, sites, out):
    pct = lambda k: 100 * pc[k].values / np.maximum(pc.valid.values, 1)
    fields = [("% of hours outside AOA (main)", pct("out_aoa")), ("% outside kNN-10 (check)", pct("out_knn")),
              ("% joint-only (outside AOA, all variables inside)", pct("out_joint_only")), ("mean DI", pc.mean_di.values)]
    fig, axes = panels(4, 4, w=4.2, h=3.6)
    for ax, (title, val) in zip(axes, fields):
        vmax = max(np.nanpercentile(val, 99), 1e-6)
        sc = _map(ax, pc.lon.values, pc.lat.values, val, sites, vmax, title)
        fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle("All variables together, per cell (dots: towers)", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "figa01_maps.png")
    plt.close(fig)


def fig_curve(curve, th, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, m, cut, title in ((axes[0], "aoa", th.aoa_cutoff, "AOA: distance to the nearest tower hour (DI)"),
                              (axes[1], "knn", th.knn_cutoff, "kNN-10: distance to the 10th nearest tower hour")):
        for g, lab, c, ls in GROUPS:
            if m == "knn" and g == "A_tower_loto":
                continue
            ax.plot(curve.di, 100 * curve[f"{g}_{m}"], color=c, ls=ls, lw=1.5, label=lab)
        ax.axvline(cut, color=INK2, lw=0.8, ls="--")
        ax.text(cut, 1, f" cut-off {cut:.3f}", fontsize=7, color=INK2, va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
        ax.set_xlim(0, min(1.0, curve.di.max()))
        ax.set_xlabel("distance (DI units: / mean distance between tower hours)")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("% of hours above")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figa02_curve.png")
    plt.close(fig)


def fig_decomposition(sh, out):
    s = sh.set_index("group")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    names = [g for g, *_ in GROUPS]
    x = np.arange(len(names))
    ax.bar(x - 0.2, 100 * s.loc[names, "outside_aoa"], width=0.38, color=[c for *_, c, _ in GROUPS],
           edgecolor="white", label="AOA (main)")
    ax.bar(x + 0.2, 100 * s.loc[names, "outside_knn"], width=0.38, color=[c for *_, c, _ in GROUPS],
           alpha=0.45, edgecolor="white", label="kNN-10 (check, lighter)")
    ax.set_xticks(x)
    ax.set_xticklabels(["towers\n(site out)", "towers\n(hour out)", "ERA5 at towers\nsame hours",
                        "ERA5 tower\ncells, all hours", "ERA5\nAustralia"], fontsize=7.5)
    ax.set_ylabel("% of hours outside")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7)
    ax.set_title("Where the gap comes from", loc="left", fontsize=10)
    ax = axes[1]
    bottom = np.zeros(len(names))
    for k in CAT_COL:
        v = 100 * s.loc[names, k].fillna(0).values
        ax.bar(x, v, bottom=bottom, color=CAT_COL[k], label=CAT_LAB[k], edgecolor="white", linewidth=0.8)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(["A site", "A hour", "B", "C", "D"])
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.set_title("Why the uncovered hours are uncovered (AOA)", loc="left", fontsize=10)
    fig.savefig(out / "figa03_decomposition.png")
    plt.close(fig)


def fig_month_hour(bm, bh, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, d, key, lab in ((axes[0], bm, "month", "month"), (axes[1], bh, "hour", "local hour")):
        ax.plot(d[key], 100 * d.outside_aoa, color=C_AUS, lw=1.6, label="AOA (main)")
        ax.plot(d[key], 100 * d.outside_knn, color=C_TOWER, lw=1.6, label="kNN-10 (check)")
        ax.set_xlabel(lab)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("% of hours outside")
    axes[0].legend(fontsize=7)
    fig.suptitle("Share of Australian hours outside, by month and by local hour", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figa04_by_month_hour.png")
    plt.close(fig)


def fig_trend(by, tr, out):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    t = tr.set_index("measure")
    for col, c, lab in (("outside_aoa", C_AUS, "AOA (main)"), ("outside_knn", C_TOWER, "kNN-10 (check)"),
                        ("outside_joint_only", "#8c3b0f", "AOA, joint only")):
        y = 100 * by[col].values
        ax.plot(by.year, y, color=c, lw=1.4, label=f"{lab}  {t.loc[col, 'trend_pp_per_decade']:+.2f} pp/decade")
        if len(by) > 2:
            ax.plot(by.year, np.polyval(np.polyfit(by.year, y, 1), by.year), color=c, lw=0.8, ls="--")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("% of hours outside")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    ax.set_title("Share of Australian weather unlike any tower hour, by year", loc="left", fontsize=10)
    fig.savefig(out / "figa05_trend.png")
    plt.close(fig)


def _site_bars(values, xlabel, title, path):
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 0.18 * len(v) + 1.0))
    ax.barh(range(len(v)), v.values, color=C_AUS, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(v.index, fontsize=7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def write_summary(info, th, sh, tr, ts, tv, out):
    s = sh.set_index("group")
    t1 = pd.DataFrame({"group": [lab for _, lab, *_ in GROUPS],
                       "outside_aoa_%": 100 * s.loc[[g for g, *_ in GROUPS], "outside_aoa"].values,
                       "outside_knn_%": 100 * s.loc[[g for g, *_ in GROUPS], "outside_knn"].values,
                       "single_%": 100 * s.loc[[g for g, *_ in GROUPS], "outside_single"].values,
                       "pair_%": 100 * s.loc[[g for g, *_ in GROUPS], "outside_pair"].values,
                       "joint_only_%": 100 * s.loc[[g for g, *_ in GROUPS], "outside_joint_only"].values,
                       "mean_DI": s.loc[[g for g, *_ in GROUPS], "mean_di"].values})
    t2 = ts.sort_values("outside_aoa_loto", ascending=False).head(10)[["site", "hours", "median_di_loto", "outside_aoa_loto"]]
    t2 = t2.assign(outside_aoa_loto=100 * t2.outside_aoa_loto)
    lines = [f"# All-variables coverage summary ({info['years'][0]}-{info['years'][1]})", "",
             f"Variables: {', '.join(SHORT.get(v, v) for v in info['variables'])}. "
             f"{th.tower_hours:,} tower hours with all of them measured ({th.sites} sites). "
             f"d_bar = {th.d_bar:.4f}. AOA cut-off (upper whisker of leave-one-tower-out DI) = {th.aoa_cutoff:.4f} "
             f"(hinges {th.loto_lower_hinge:.4f} / {th.loto_upper_hinge:.4f}). "
             f"kNN-10 cut-off (99.8th percentile of the towers' own 10th-neighbour distance) = {th.knn_cutoff:.4f}.", "",
             "## Share of hours outside (%)", "", md_table(t1, "{:.3f}"), ""]
    if len(tr):
        lines += ["## Trend (Australia)", "", md_table(tr, "{:.4f}"), ""]
    lines += ["## Most unusual towers (share of own hours outside when the site is left out, %)", "",
              md_table(t2, "{:.3f}"), ""]
    if tv is not None:
        t3 = tv.sort_values("increase", ascending=False).head(10).assign(increase=lambda d: 100 * d.increase)[["site", "hours", "increase"]]
        lines += ["## Most valuable towers (extra % of Australia outside AOA without the site; sample)", "",
                  md_table(t3, "{:.3f}"), ""]
    (out / "summary_aoa.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_aoa")
    p.add_argument("--towers", default="data")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((res / "aoa_run_info.json").read_text())
    th = pd.read_csv(res / "aoa_threshold.csv").iloc[0]
    sh = pd.read_csv(res / "aoa_shares.csv")
    curve = pd.read_csv(res / "aoa_curve.csv")
    by, bm, bh = (pd.read_csv(res / f"aoa_by_{k}.csv") for k in ("year", "month", "hour"))
    tr = pd.read_csv(res / "aoa_trend.csv")
    ts = pd.read_csv(res / "aoa_tower_sites.csv")
    tv = pd.read_csv(res / "aoa_tower_value.csv") if (res / "aoa_tower_value.csv").exists() else None
    sites = pd.read_csv(Path(a.towers) / "tower_sites.csv")
    pc = xr.open_dataset(res / "aoa_per_cell.nc")

    fig_maps(pc, sites, out)
    fig_curve(curve, th, out)
    fig_decomposition(sh, out)
    fig_month_hour(bm, bh, out)
    fig_trend(by, tr, out)
    _site_bars(100 * ts.set_index("site").outside_aoa_loto, "% of the site's own hours outside AOA",
               "How unlike the other towers each tower is (left out in turn)", out / "figa06_tower_sites.png")
    if tv is not None:
        _site_bars(100 * tv.set_index("site").increase, "extra % of Australian hours outside AOA",
                   "Value of each tower (leave it out, see how much more weather is uncovered)",
                   out / "figa07_tower_value.png")
    write_summary(info, th, sh, tr, ts, tv, out)
    print(f"figures and summary_aoa.md -> {out}")


if __name__ == "__main__":
    main()
