# Isotomics-Automated

`isotomics` provides tools for M+1 Orbitrap-IRMS workflows:
- simulate expected precision for proposed experiments
- process experimental `.isox` data to estimate site-specific isotope values

## Install

```bash
pip install isotomics
```

## Quickstart

Run the packaged example workflow:

```bash
isotomics-quickstart
```

or

```bash
python -m isotomics
```

The package bundles example inputs under `isotomics/input_data/` and quickstart reads them directly.
Output files are written to your current working directory.

## Input CSV Format

Each row describes one constrained site.

Required columns:
- `Site Names`
- `Element`
- `Number Atoms`

Fragment columns:
- add one or more columns named like `Fragment 44`, `Fragment 133`, etc.
- use `1` if the site is present in that fragment and `0` if absent

See the bundled example:
- `isotomics/input_data/Example_Molecule_Input.csv`

## Core Workflows

1. Simulation:
- build a molecule definition from the input CSV
- simulate M+N observations from user-provided delta values
- solve and visualize expected site-specific outputs

2. Experimental processing:
- read processed `.isox` files grouped by fragment and `Smp`/`Std`
- compute isotope ratios and uncertainties
- solve for site-specific values using the M+1 Monte Carlo pipeline

## Example Dataset

The alanine example dataset is bundled under:
- `isotomics/input_data/Experimental_Data/`

This is the dataset used by the quickstart command.

## Citation

Csernica, Timothy and Zeichner, Sarah S. (2026).  
Csernica/isotomics.


