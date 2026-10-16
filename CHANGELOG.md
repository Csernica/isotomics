# Changelog

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
