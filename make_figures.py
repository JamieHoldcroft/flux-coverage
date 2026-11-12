"""
make_figures.py
===============

Step 4 (laptop or Gadi). Turns the coverage_stats.py results folder into
figures and a short summary. Needs only numpy, pandas, xarray, matplotlib.

    python make_figures.py --results results --out figures

Figures (in --out):
    fig01_distributions.png     tower vs ERA5 distributions (log density; rain as share of hours above x)
    fig02_map_outside.png       per cell: % of hours outside the tower range
    fig03_map_below_above.png   the same split into below (cold/dry/calm side) and above
    fig04_ranges.png            spreads side by side: tower, matched ERA5, tower cells, Australia
    fig05_decomposition.png     % outside the tower range: matched -> tower cells -> Australia
    fig06_by_month.png          % outside by month
    fig07_by_hour.png           % outside by local hour
    fig08_trend.png             % outside by year, 1980-2025, with linear trend
    fig09_tower_value.png       how much the uncovered share grows when each site is left out
    fig10_point_vs_grid.png     tower vs ERA5 at its own cell: r and bias per site
    summary.md                  the key numbers as tables
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = ["tair", "vpd", "qair", "psurf", "swdown", "lwdown", "precip", "wind"]
QUANTITIES = VARIABLES
LABEL = {"tair": "Air temperature (°C)", "vpd": "VPD (hPa)", "qair": "Specific humidity (g/kg)",
         "psurf": "Surface pressure (hPa)", "swdown": "Shortwave down (W/m²)",
         "lwdown": "Longwave down (W/m²)",
         "precip": "Rain (mm/h)", "wind": "Wind speed (m/s)"}
SHORT = {"tair": "Tair", "vpd": "VPD", "qair": "Qair", "psurf": "Psurf", "swdown": "SW",
         "lwdown": "LW", "precip": "Rain", "wind": "Wind"}
LOGX = {"precip"}
MAIN, OWN = "tower_p0.1-99.9", "matched_p0.1-99.9"

# colours (validated categorical slots 1-3 and a one-hue sequential ramp)
C_TOWER, C_AUS, C_CELLS, C_MATCH = "#222222", "#1aa7a0", "#9a9a9a", "#7b61d9"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Oranges"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})


def panels(n, ncol, w=3.3, h=2.5):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def rebin(h, edges, display):
    """Fine histogram -> density on display bins (fine bins assigned by centre)."""
    e = edges.copy()
    finite = np.isfinite(e)
    e[~finite] = np.nan
    centre = 0.5 * (e[:-1] + e[1:])
    idx = np.searchsorted(display, centre, side="right") - 1
    ok = np.isfinite(centre) & (idx >= 0) & (idx < len(display) - 1)
    out = np.bincount(idx[ok], weights=h[ok], minlength=len(display) - 1)
    total = h.sum()
    return out / total / np.diff(display) if total > 0 else out


def fig_distributions(hs, thr, ranges, out):
    fig, axes = panels(len(QUANTITIES), 2, w=4.2, h=2.6)
    for ax, q in zip(axes, QUANTITIES):
        e = hs[f"{q}_edges"].values
        r = ranges[ranges.quantity == q].set_index("group")
        lo_x = min(r.loc[g, "min"] for g in ("A_tower", "D_australia"))
        hi_x = max(r.loc[g, "max"] for g in ("A_tower", "D_australia"))
        if q in LOGX:
            lo_x = max(lo_x, 0.01)
            display = np.geomspace(lo_x, hi_x * 1.01, 90)
        else:
            display = np.linspace(lo_x, hi_x + 1e-9, 120)
        series = [("ERA5, all Australia", hs[f"{q}_D"].values, C_AUS, "-"),
                  ("ERA5, tower cells 1980-2025", hs[f"{q}_C"].values, C_CELLS, "--"),
                  ("Towers (measured hours)", hs[f"{q}_A"].values, C_TOWER, "-")]
        mid = np.sqrt(display[:-1] * display[1:]) if q in LOGX else 0.5 * (display[:-1] + display[1:])
        for name, h, c, ls in series:
            if q in LOGX:
                # rain: share of hours ABOVE x (gauge tips make a density spiky)
                above = np.concatenate([np.cumsum(h[::-1])[::-1], [0.0]]) / h.sum()   # share >= e[i]
                xs = display
                d = above[np.searchsorted(e, xs)]
                if name.startswith("Towers"):
                    ax.fill_between(xs, d, color=c, alpha=0.25, lw=0)
                ax.plot(xs, d, color=c, lw=1.6 if ls == "-" else 1.2, ls=ls, label=name)
                continue
            d = rebin(h, e, display)
            if name.startswith("Towers"):
                ax.fill_between(mid, d, color=c, alpha=0.25, lw=0, step="mid")
            ax.step(mid, d, where="mid", color=c, lw=1.6 if ls == "-" else 1.2, ls=ls, label=name)
        lo, hi = thr[(MAIN, q)]
        for v in (lo, hi):
            if (not q in LOGX) or v > 0:
                ax.axvline(v, color=INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        if q in LOGX:
            ax.set_xscale("log")
        ax.set_title(LABEL[q], loc="left", fontsize=9, color=INK)
        ax.set_ylabel("share of hours above x" if q in LOGX else "density (log)")
    axes[0].legend(loc="lower center", fontsize=7.5)
    fig.suptitle("What the towers measured vs all Australian weather (ERA5 hourly)  ·  dotted = tower range (0.1–99.9 %)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig01_distributions.png")
    plt.close(fig)


def _grid(lon, lat, val):
    """Cell values -> 2D grid (for pcolormesh), at the cells' own spacing."""
    ulon, ulat = np.unique(lon), np.unique(lat)
    step = np.min(np.diff(ulon)) if len(ulon) > 1 else 0.25
    glon = np.arange(ulon.min(), ulon.max() + step / 2, step)
    glat = np.arange(ulat.min(), ulat.max() + step / 2, step)
    g = np.full((len(glat), len(glon)), np.nan)
    g[np.round((lat - glat[0]) / step).astype(int), np.round((lon - glon[0]) / step).astype(int)] = val
    return np.append(glon - step / 2, glon[-1] + step / 2), np.append(glat - step / 2, glat[-1] + step / 2), g


def _map(ax, lon, lat, val, sites, vmax, title):
    ex, ey, g = _grid(lon, lat, val)
    sc = ax.pcolormesh(ex, ey, g, cmap=SEQ, vmin=0, vmax=vmax, shading="flat")
    ax.scatter(sites.lon, sites.lat, s=7, color=INK, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_aspect(1 / np.cos(np.radians(27)))
    ax.set_xlim(112, 154.5)
    ax.set_ylim(-44.5, -9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9)
    return sc


def fig_maps(pc, sites, out):
    lon, lat = pc.lon.values, pc.lat.values
    fig, axes = panels(len(QUANTITIES), 4, w=3.0, h=2.6)
    for ax, q in zip(axes, QUANTITIES):
        valid = pc[f"{q}_valid"].values.sum(axis=0)
        outside = pc[f"{q}_below"].values.sum(axis=0) + pc[f"{q}_above"].values.sum(axis=0)
        share = np.where(valid > 0, 100 * outside / np.maximum(valid, 1), np.nan)
        vmax = max(1.0, np.nanpercentile(share, 98))
        sc = _map(ax, lon, lat, share, sites, vmax, SHORT[q])
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label("% of hours", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Share of hours outside the range the towers measured (dots = towers)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig02_map_outside.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(QUANTITIES), 2, figsize=(6.2, 2.6 * len(QUANTITIES)), squeeze=False)
    for row, q in zip(axes, QUANTITIES):
        valid = pc[f"{q}_valid"].values.sum(axis=0)
        for ax, side in zip(row, ("below", "above")):
            n = pc[f"{q}_{side}"].values.sum(axis=0)
            share = np.where(valid > 0, 100 * n / np.maximum(valid, 1), np.nan)
            vmax = max(1.0, np.nanpercentile(share, 98))
            sc = _map(ax, lon, lat, share, sites, vmax, f"{SHORT[q]}: {side} the tower range")
            cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
            cb.set_label("% of hours", fontsize=7)
            cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig03_map_below_above.png")
    plt.close(fig)


def fig_ranges(ranges, out):
    groups = [("A_tower", "Towers", C_TOWER), ("B_matched", "ERA5 at tower, same hours", C_MATCH),
              ("C_tower_cells", "ERA5 at tower cells, 1980-2025", C_CELLS), ("D_australia", "ERA5, all Australia", C_AUS)]
    fig, axes = panels(len(QUANTITIES), 2, w=4.2, h=1.7)
    for ax, q in zip(axes, QUANTITIES):
        r = ranges[ranges.quantity == q].set_index("group")
        for i, (g, name, c) in enumerate(groups[::-1]):
            x = r.loc[g]
            ax.plot([x["min"], x["max"]], [i, i], color=c, lw=0.8, alpha=0.6)
            ax.plot([x["p0.1"], x["p99.9"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
            ax.plot([x["p25"], x["p75"]], [i, i], color=c, lw=7, solid_capstyle="butt", alpha=0.85)
            ax.plot(x["p50"], i, "o", color="white", ms=3.5, mec=c, mew=1.2)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[1] for g in groups[::-1]], fontsize=7)
        if q == "precip":
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(LABEL[q], loc="left", fontsize=9)
    fig.suptitle("Spread of each variable: thin = min–max, line = 0.1–99.9 %, bar = 25–75 %, dot = median",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig04_ranges.png")
    plt.close(fig)


def fig_decomposition(shares, out):
    s = shares[shares.range == MAIN].pivot(index="quantity", columns="group", values="outside").loc[QUANTITIES] * 100
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    x = np.arange(len(QUANTITIES))
    for k, (g, name, c) in enumerate([("B_matched", "ERA5 at the towers, same hours", C_MATCH),
                                      ("C_tower_cells", "ERA5 at the tower cells, all hours 1980-2025", C_CELLS),
                                      ("D_australia", "ERA5, all Australia 1980-2025", C_AUS)]):
        ax.bar(x + (k - 1) * 0.27, s[g].values, width=0.25, color=c, label=name, edgecolor="white", linewidth=1)
    ax.axhline(0.2, color=INK2, lw=0.8, ls=":")
    ax.text(len(QUANTITIES) - 0.5, 0.2, " towers themselves (≤0.2 % by definition)", va="bottom", ha="right", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[q] for q in QUANTITIES])
    ax.set_ylabel("% of hours outside\nthe tower range")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3)
    ax.set_title("Where the gap comes from: point vs grid  →  which years  →  which places", loc="left", fontsize=10)
    fig.savefig(out / "fig05_decomposition.png")
    plt.close(fig)


def fig_lines(df, key, xlabel, out, name, title):
    fig, axes = panels(len(QUANTITIES), 4, w=2.6, h=2.0)
    for ax, q in zip(axes, QUANTITIES):
        d = df[(df.range == MAIN) & (df.quantity == q)].sort_values(key)
        ax.plot(d[key], 100 * d["above"], color=C_AUS, lw=1.6, label="above")
        ax.plot(d[key], 100 * d["below"], color=C_TOWER, lw=1.6, label="below")
        ax.set_title(SHORT[q], loc="left", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("% of hours")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def fig_trend(by_year, trend, out):
    fig, axes = panels(len(QUANTITIES), 4, w=2.6, h=2.0)
    for ax, q in zip(axes, QUANTITIES):
        d = by_year[(by_year.range == MAIN) & (by_year.quantity == q)].sort_values("year")
        t = trend[(trend.range == MAIN) & (trend.quantity == q)].iloc[0]
        y = 100 * d["outside"].values
        ax.plot(d.year, y, color=C_AUS, lw=1.4)
        if len(d) > 2:
            fit = np.polyfit(d.year, y, 1)
            ax.plot(d.year, np.polyval(fit, d.year), color=INK2, lw=0.9, ls="--")
        ax.set_title(f"{SHORT[q]}  {t['trend_pp_per_decade']:+.2f} pp/decade", loc="left", fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
    axes[0].set_ylabel("% of hours outside")
    fig.suptitle("Share of Australian weather outside the tower range, by year", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig08_trend.png")
    plt.close(fig)


def fig_tower_value(tv, out):
    m = tv.pivot(index="site", columns="quantity", values="increase")[QUANTITIES] * 100
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.5, 0.2 * len(m) + 1.2))
    vmax = max(0.05, np.nanpercentile(m.values, 99))
    im = ax.imshow(m.values, aspect="auto", cmap=SEQ, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(QUANTITIES)))
    ax.set_xticklabels([SHORT[q] for q in QUANTITIES], rotation=45, ha="right")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("extra % of Australian hours\nuncovered without this site", fontsize=7)
    ax.set_title("Value of each tower (leave it out, see how much more weather is uncovered)", loc="left", fontsize=9)
    fig.savefig(out / "fig09_tower_value.png")
    plt.close(fig)


def fig_point_vs_grid(pvg, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.2 * pvg.site.nunique() + 1.5))
    for ax, col, cmap, title in ((axes[0], "r", "Blues", "Correlation r (hourly)"),
                                 (axes[1], "bias", "RdBu_r", "Bias ERA5 − tower, in sd of the tower")):
        m = pvg.pivot(index="site", columns="variable", values=col)[VARIABLES]
        if col == "bias":
            sd = pvg.pivot(index="site", columns="variable", values="tower_sd")[VARIABLES]
            m = m / sd
            im = ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5)
        else:
            im = ax.imshow(m.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(VARIABLES)))
        ax.set_xticklabels([SHORT[v] for v in VARIABLES], rotation=45, ha="right")
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index, fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(title, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig10_point_vs_grid.png")
    plt.close(fig)


def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(ranges, shares, trend, tv, pvg, out):
    D = shares[shares.group == "D_australia"]
    t1 = D[D.range == MAIN].set_index("quantity").loc[QUANTITIES][["below", "above", "outside"]] * 100
    t1.insert(0, "quantity", [SHORT[q] for q in QUANTITIES])
    sens = D.pivot(index="quantity", columns="range", values="outside").loc[QUANTITIES]
    sens = sens[[MAIN, "tower_p1-99", "tower_min-max", "halfhourly_p0.1-99.9", OWN, "towercells_p0.1-99.9"]] * 100
    sens.insert(0, "quantity", [SHORT[q] for q in QUANTITIES])
    dec = shares[shares.range == MAIN].pivot(index="quantity", columns="group", values="outside").loc[QUANTITIES] * 100
    dec.insert(0, "quantity", [SHORT[q] for q in QUANTITIES])
    rr = ranges[ranges.group.isin(["A_tower", "D_australia"])].copy()
    rr = rr[["group", "quantity", "min", "p0.1", "p25", "p50", "p75", "p99.9", "max", "sd"]]
    tr = trend[trend.range == MAIN][["quantity", "share_first_decade", "share_last_decade", "trend_pp_per_decade",
                                     "trend_below_pp_per_decade", "trend_above_pp_per_decade"]].copy()
    tr[["share_first_decade", "share_last_decade"]] *= 100
    top = tv.sort_values("increase", ascending=False).groupby("quantity").head(3)
    top = top[["quantity", "site", "increase"]].assign(increase=lambda d: d.increase * 100)
    wet = ranges[(ranges.quantity == "precip")][["group", "wet_share"]].assign(wet_share=lambda d: d.wet_share * 100)
    med = pvg.groupby("variable")[["r", "bias", "sd_ratio"]].median().loc[VARIABLES].reset_index()
    text = f"""# Single-variable coverage, Australia (hourly)

Share of Australian land-hours (ERA5, area weighted) outside the tower range
(0.1 to 99.9 percentiles of every fully measured tower hour), %:

{md_table(t1)}

## Sensitivity to how the tower range is defined (% outside)

{md_table(sens)}

## Where the gap comes from (% outside the main tower range)

A = towers, B = ERA5 at the towers at the same hours, C = ERA5 at the tower
cells for all hours, D = all Australia.

{md_table(dec)}

## Spreads (tower A vs Australia D)

{md_table(rr)}

## Wet hours (>= 0.2 mm/h), % of all hours

{md_table(wet)}

## Trend of the share outside (pp = percentage points)

{md_table(tr, "{:.3f}")}

## Most valuable towers (extra % uncovered when left out, top 3 per quantity)

{md_table(top, "{:.3f}")}

## Tower vs ERA5 at its own cell (median over sites)

{md_table(med)}
"""
    (out / "summary.md").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ranges = pd.read_csv(res / "ranges.csv")
    thr_df = pd.read_csv(res / "thresholds.csv")
    thr = {(r.range, r.quantity): (r.lo, r.hi) for r in thr_df.itertuples()}
    shares = pd.read_csv(res / "shares.csv")
    by_year = pd.read_csv(res / "shares_by_year.csv")
    by_month = pd.read_csv(res / "shares_by_month.csv")
    by_hour = pd.read_csv(res / "shares_by_hour.csv")
    trend = pd.read_csv(res / "trend.csv")
    tv = pd.read_csv(res / "tower_value.csv")
    pvg = pd.read_csv(res / "point_vs_grid.csv")
    sites = pd.read_csv(res / "sites_cells.csv")
    hs = xr.open_dataset(res / "histograms.nc")
    pc = xr.open_dataset(res / "per_cell.nc")

    fig_distributions(hs, thr, ranges, out)
    fig_maps(pc, sites, out)
    fig_ranges(ranges, out)
    fig_decomposition(shares, out)
    fig_lines(by_month, "month", "month", out, "fig06_by_month.png", "Share of hours outside the tower range, by month")
    fig_lines(by_hour, "hour", "local hour", out, "fig07_by_hour.png", "Share of hours outside the tower range, by local hour")
    fig_trend(by_year, trend, out)
    fig_tower_value(tv, out)
    fig_point_vs_grid(pvg, out)
    write_summary(ranges, shares, trend, tv, pvg, out)
    print(f"figures and summary.md -> {out}")


if __name__ == "__main__":
    main()
