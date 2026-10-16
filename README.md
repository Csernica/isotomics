# Isotomics

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

Current isotope support is hardcoded in both `isotomics/basicDeltaOperations.py` and `isotomics/fragmentAndSimulate.py`.
Available elements are currently: `C`, `N`, `H`, `O`, and `S`.

Required columns:
- `Site Names`
- `Element`
- `Number Atoms`

Fragment columns:
- add one or more columns named like `Fragment 44`, `Fragment 133`, etc.
- use `1` if the site is present in that fragment and `0` if absent
- if a fragment forms by more than one pathway, give one column per pathway (`Fragment 82_01`, `Fragment 82_02`)
  and add a final row named `Relative Contribution` with each pathway's share (summing to 1 per fragment)

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

If you use `isotomics`, please cite the method papers:

- Csernica, T. and Eiler, J. M. (2023). High-dimensional isotomics, part 1: A mathematical framework for isotomics. *Chemical Geology*, 617, 121235. [doi:10.1016/j.chemgeo.2022.121235](https://doi.org/10.1016/j.chemgeo.2022.121235)
- Weiss, G. M., Sessions, A. L., Julien, M., Csernica, T., Yamada, K., Gilbert, A., Freeman, K. H., and Eiler, J. M. (2023). Analysis of intramolecular carbon isotope distributions in alanine by electrospray ionization Orbitrap mass spectrometry. *International Journal of Mass Spectrometry*, 493, 117128. [doi:10.1016/j.ijms.2023.117128](https://doi.org/10.1016/j.ijms.2023.117128) (source of the bundled alanine example dataset)

and the software:

- Csernica, T. and Zeichner, S. S. (2026). isotomics. https://github.com/Csernica/isotomics

## Documentation

Full documentation, including tutorials and the API reference, is at https://isotomics.readthedocs.io.

To build the docs locally:

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```
