# flux-coverage

How much of Australia's weather has the flux tower network actually measured?
One variable at a time, hourly: the range of every fully measured hour at the
42 Australian towers, against every hour of ERA5 at every Australian land cell,
1980-2025.

## Run order

| Step | Where | Command | Output |
|---|---|---|---|
| 1 | laptop | `python tower_hourly.py --met C:/Users/jamie/CCRC/FLUXNET/Met --out data` (~3 min) | `data/tower_*.csv*` |
| 2 | Gadi login node | `python3 era5_hourly_australia.py --make-cells --out /scratch/dt6/$USER/fc/era5_hourly` (seconds) | `cells.csv` |
| 3 | Gadi | one-year test, then the full pull (see below) | `era5_hourly_<year>.nc` |
| 4 | Gadi | two-year stats test + `check_coverage_stats.py`, then the full stats run | `results/` + `results/figures/` |
| 5 | Gadi | pairs: two-year test + `check_coverage_pairs.py`, then the full run | `results_pairs/` + figures |
| 6 | laptop (optional) | `python make_figures.py --results results --out figures` (and `make_figures_pairs.py`) | figures, summaries |

Gadi commands (from the repo folder on Gadi, with `data/` copied next to the scripts):

```bash
module unload python3; module load python3/3.11.7
python3 era5_hourly_australia.py --make-cells --out /scratch/dt6/$USER/fc/era5_hourly
# step 3a: two test years, small job (<= 8 SU). Read the time per year from the log.
qsub -N era5h_test -l ncpus=4,mem=32GB,walltime=01:00:00 -v START=2010,END=2011,WORKERS=2 run_era5_hourly.pbs
# step 3b: everything else (years already done are skipped)
qsub run_era5_hourly.pbs
# step 4a: stats on the two test years + brute-force self-check (the log must say ALL PASS)
qsub -N cov_test -l walltime=01:00:00 -v START=2010,END=2011,OUT=results_test,CHECK=1 run_coverage_stats.pbs
# step 4b: the real run, 1980-2025
qsub run_coverage_stats.pbs
# step 5a: pairs on two test years + brute-force self-check (the log must say ALL PASS)
qsub -N pairs_test -l ncpus=16,mem=64GB,walltime=01:30:00 -v START=2010,END=2011,WORKERS=2,OUT=results_pairs_test,CHECK=1 run_coverage_pairs.pbs
# step 5b: the real pairs run, 1980-2025
qsub run_coverage_pairs.pbs
```

`check_coverage_stats.py` recounts every Australia share the slow, obvious way
straight from the ERA5 files and compares (exact to 1e-8), checks the per-cell
counts, and checks that the quantiles are within one fine bin.
`check_coverage_pairs.py` does the same for the pairs: it rebuilds every
envelope with its own code and recounts every share, combination-only share,
per-cell count, yearly share and leave-one-site-out share.

## Method in one paragraph

Tower side: every half-hour flagged measured (qc = 0) in the tower met files,
after removing values outside physical bounds and nine known faulty sensor
stretches (listed in `tower_hourly.py`), averaged to UTC hours that are fully
measured. ERA5 side: hourly 2 m temperature, dewpoint, surface pressure,
10 m wind, shortwave, longwave and rain at every Australian 0.25° land cell,
turned into the same variables and units and the same hour definition (hour
starting at the stamp). The tower range is the 0.1 to 99.9 percentiles of the
pooled tower hours; the headline is the area-weighted share of Australian
land-hours outside it. Groups A (towers), B (ERA5 at the towers, same hours),
C (ERA5 at the tower cells, all hours) and D (all Australia) split the gap
into point-vs-grid, which-years and which-places parts.

## Choices worth knowing

* Hourly, not half-hourly: ERA5 is hourly. The raw half-hourly tower range is
  kept as a sensitivity (`halfhourly_p0.1-99.9`).
* Rain is compared as a rate (mm/h), all hours. The share of wet hours
  (at least 0.2 mm/h, about one tip of a tower gauge) is reported alongside.
* Qair in the tower files is derived from VPD and Tair, so it counts as
  measured only when both are.
* Tower wind is at the measurement height (2-95 m), ERA5 wind at 10 m. Not
  corrected; the point-vs-grid table shows the size of the offset.
* Pairs (`coverage_pairs.py`): the 2-D tower envelope is a grid of 40 bins
  across each variable's tower range (rain on a log axis), minus the sparsest
  bins holding 0.2 % of tower hours. Rain is rounded to the gauges' 0.2 mm/h
  steps for the pairs, so ERA5 is not penalised for values between two gauge
  steps. "Combination-only" = outside the envelope but inside both 1-variable
  ranges. Variants: 20/40/80 bins x 0/0.2/1 % trimmed.
* Shares of Australia are exact (bin edges include every range end); its
  quantiles are interpolated within fine bins (e.g. 0.05 °C).
