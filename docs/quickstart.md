# Quickstart

The quickstart runs the complete M+1 experimental workflow on a real dataset that ships with
the package: Orbitrap measurements of an alanine sample and standard (Weiss et al., 2023). It
takes about ten seconds and is useful for two things:

- **Checking your installation.** The results are deterministic, so your output should match the
  values below exactly.
- **Seeing what the pipeline produces** before you set it up for your own data.

## Run it

Run it from an empty folder, because it writes its output files to the current directory and
overwrites any existing files with the same names:

```bash
mkdir isotomics-quickstart && cd isotomics-quickstart
isotomics-quickstart
```

`python -m isotomics` does the same thing.

## What it does

1. Reads the alanine molecule definition (`Example_Molecule_Input.csv`): six sites, observed in
   the intact molecule and in fragment 44.
2. Processes 25 `.isox` files from three measurement folders (fragment 44 M+1 relative
   abundances, full-molecule M+1 relative abundances and full-molecule isotope ratios), keeping
   scans from 5 to 15 minutes and screening each file for peak drift, zero-intensity scans and
   poor signal-to-noise.
3. Solves for the sample's site-specific δ values with the M+1 Monte Carlo routine, using the
   standard's known carbon values (−11.9 and −9.95 ‰) and the sample's molecular-average
   δ¹³C (−0.8 ± 0.1 ‰, from an external measurement).

The terminal shows each folder as it is processed (with the files it read), then:

```text
Writing experimental results to CSV: M1Output.csv
CSV write complete.
Saved plot: M1Output.png
```

It writes three files: `all_data_output.csv`, `M1Output.csv` and `M1Output.png`.

You will also see a warning that the deuterium label `D` was renamed to `2H`. This is expected:
the bundled raw `.isox` files predate the `2H` label, and isotomics renames it when reading them.

## Output files

### `all_data_output.csv`: processed measurements

One row per file and per measured quantity (91 rows), for example the ¹³C M+1 relative
abundance of fragment 44 in one sample acquisition. The main columns are:

| Column | Meaning |
| --- | --- |
| `FileName`, `Fragment`, `File Type` | Source file, measurement folder, and `Smp` or `Std` |
| `MN Relative Abundance` / `IsotopeRatio` | The quantity: a substitution (e.g. `13C`) for M+1 relative abundances, or a ratio (e.g. `13C/Unsub`) |
| `Average`, `StdDev`, `StdError`, `RelStdError` | Mean over scans and its uncertainty |
| `ShotNoise`, `RSE over SN` | Shot-noise limit and the ratio of observed error to it (used in screening) |
| `Tic`, `TicVar` | Total ion current and its variation |

Use it to check individual acquisitions and replicate agreement.

### `M1Output.csv`: site-specific results

One row per site, labeled by the `Site Names` from the molecule CSV. You should get:

| Site Names | Element (atoms) | Relative Deltas (‰) | CRF Deltas (‰) |
| --- | --- | --- | --- |
| Calphabeta | C (2) | −0.151 ± 0.399 | −12.050 ± 0.394 |
| Ccarboxyl | C (1) | 31.961 ± 0.782 | 21.693 ± 0.774 |
| Ocarboxyl | O (2) | −0.591 ± 15.617 | −0.591 ± 15.617 |
| Namine | N (1) | −0.711 ± 1.054 | −0.711 ± 1.054 |
| Hretained | H (6) | 1.046 ± 1.257 | 1.046 ± 1.257 |
| Hlost | H (2) | 3.724 ± 14.563 | 3.724 ± 14.563 |

- **Relative Deltas** are the sample relative to the standard. This sample is enriched by about
  32 ‰ in the carboxyl carbon and is otherwise indistinguishable from the standard within error.
- **CRF Deltas** are the same results in the reference frame of the standard values supplied to
  the solver (`Standard Delta Values in CRF`). Only the carbon sites were known for this
  standard, so the other sites are 0 there and their CRF values equal their relative values.
- Large errors on the oxygen and lost hydrogen sites are expected: only the intact molecule
  constrains them.

### `M1Output.png`: results plot

The relative δ values and their errors for each site, as in the table above.

## Next steps

The {doc}`tutorials/index` walk through the same workflow step by step, explain each input,
and show how to set it up for your own molecule and data.
