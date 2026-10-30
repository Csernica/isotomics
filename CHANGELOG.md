# Changelog

## 0.2.4 (2026-10-30)

### Fixed

- **The alanine standard's carbon values were assigned to the wrong sites, which changes results.**
  The natural-abundance standard (Weiss et al., 2023, Table 2) has −11.9 ‰ for the carboxyl carbon
  (C-1) and −12.9 and −7.0 ‰ for C-2 and C-3, so −9.95 ‰ for the α/β site. The quickstart, example
  notebook and tests passed `[-11.9, -9.95, ...]` in the site order Calphabeta, Ccarboxyl; they now
  pass `[-9.95, -11.9, ...]`. Relative δ values of the quickstart sample change from −0.151 to
  −0.785 ‰ (Calphabeta) and 31.960 to 31.321 ‰ (Ccarboxyl), in line with the Orbitrap values of
  Weiss et al. (−0.8 and 32.0 ‰); values in the standard's reference frame change by about 1.3–2.6 ‰.
  If you used these standard values in your own scripts, check their order.

### Changed

- `experimentalDataM1` prints the full paths of the results CSV and the plot it writes.
  `resultsFileName=None` skips writing the CSV.

### Documentation

- The experimental tutorial compares the results with Weiss et al. (2023) and explains how to set up
  your own data.

## 0.2.3 (2026-10-28)

### Fixed

- **Errors of replicate averages, which changes results.** `rtnMeans` divided the errors of the mean
  by the square root of a single file count for every group, rather than counting the sample and
  standard files of each quantity separately. With 3 sample and 4 or 6 standard acquisitions, as in
  the bundled data, the standard's errors were overstated by 1.15× and 1.41×. δ values are
  essentially unchanged, but their errors shrink (e.g. carboxyl carbon 31.961 ± 0.782 →
  31.960 ± 0.748 ‰).
- The shot-noise limit of M+N relative abundances now follows eq. S.12 of the supplement to
  Csernica et al. (High-dimensional isotomics, part 2). The old formula overestimated it, most for
  the dominant peak (RSE/SN for ¹³C in fragment 44 was 0.61, now 1.40). This affects the
  `ShotNoise` and `RSE over SN` columns and the RSE/SN screen, not the solver.
- The bundled processed-data table (`Experimental_Data_Input.csv`) was regenerated with the
  corrected errors, so solving from it matches processing the raw files.

### Changed

- In `rtnMeans`, the `IsotopeRatio` column is next to `MN Relative Abundance`.

## 0.2.2 (2026-10-27)

### Fixed

- Processing silently dropped scans in which the alphabetically first peak (e.g. ¹³C) was not
  detected: their time was recorded as 0, so the time window excluded them. Scans are now kept as
  long as any peak was observed. Results for the bundled data are unchanged.
- `.isox` files containing more than one compound now raise an error. Peaks are identified by name
  only, so several fragments in one file were silently mixed together; each compound should be
  exported to its own file and folder.
- `peakDriftScreen` warnings named the last peak in the list rather than the one that drifted, and
  converted the drift to ppm with the wrong mass.
- `RSESNScreen` ignored its threshold argument (always 2).

### New

- `RSESNThreshold` (default 2) in `processIndividualAndAverageIsotopeRatios`, alongside the other
  screening thresholds, which are now documented.
- The scan tables in `rtnMergedDict` no longer include a meaningless `isotopolog` column.
- The experimental tutorial covers the data layout, processing and the screening options.

## 0.2.1 (2026-10-19)

### Fixed

- `fullSpectrumVis` left out the unsubstituted (`Unsub`) peak, so it was missing from the plot and
  every other peak was normalized against the wrong total.
- The unsubstituted molecular-average peak was unlabeled in plots and in IsoX peak lists written with
  `outputIsotopologs=True`; it is now `Unsub`, matching `.isox` data.
- `unresolvedDict` in the documented format (`{'M1': {'full': {'17O': '13C'}}}`) was silently
  ignored by `trackMNFragments`.
- `simulateSmpStd` returned all-`NaN` results when the U value named by `UMNSub` was not simulated;
  it now raises an error explaining how to fix it.

### New

- `logY` option for `fullSpectrumVis` and `MNSpectrumVis`.
- `simulateSmpStd` accepts `unresolvedDict`.
- The simulation tutorial covers unresolved and unobserved peaks, `UMNSub` and `perturbTheoryOAmt`.

## 0.2.0 (2026-10-16)

### Breaking changes

- **Errors are given as fractions everywhere** (0.001 = 1 ‰). Error CSVs (`errorPath`) and
  `mARelErr` in `experimentalDataM1` were previously in per mil; divide existing values by 1000.
  `MNError` and `UValueError` in `simulateSmpStd` were already fractions and are unchanged, as are
  `perturbTheoryOAmt` and `ffstd`.
- **Deuterium is labeled `2H`** instead of `D`, like the other isotopes (`13C`, `15N`, …). This
  affects substitution names, peak and column names (`2H-13C`, `2H/Unsub`, `counts2H`) and the
  bundled example files. Older input using `D` still works: it is renamed to `2H`, with a one-time
  warning, when reading `.isox` files, error CSVs and processed tables, and in substitution names
  passed to functions (`omitMeasurements`, `UValue`, `targetRatio`, …).

### New

- Fragments formed by more than one pathway: give one `Fragment 82_01`, `Fragment 82_02`, …
  column per pathway and a final `Relative Contribution` row in the molecule CSV.
- The M+1 results CSV (and the simulation results CSV) now starts with a `Site Names` column.
- `experimentalDataM1(plotPath=...)` saves the results plot; `isotomics-quickstart` writes
  `M1Output.png`.
- `simulateSmpStd` warns when `MNError`/`UValueError` are passed together with `errorPath`, which
  takes precedence.
- Tutorials on ReadTheDocs for simulating an experiment and processing experimental data, and a
  fuller quickstart page.

### Fixed

- Compatibility with pandas 3 and numpy 2.
- Fragments with more than one subgeometry were silently dropped when read from the molecule CSV,
  and crashed the solver.
- Simulations failed with a `KeyError` after an incomplete rename of the U value key.

### Other

- Dependencies are capped below their next major versions; Python 3.10 or later.
- Project links (documentation, source, issues) on PyPI.
- The unused standard delta columns were removed from the example molecule CSV.

## 0.1.3 (2026-03-05) and earlier

Initial releases.
