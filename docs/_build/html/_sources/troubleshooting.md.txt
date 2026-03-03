# Troubleshooting

## `No module named isotomics`

Make sure the active environment has the package installed:

```bash
pip install isotomics
```

## `No module named pytest`

Install test dependencies:

```bash
pip install -e .[test]
```

## Quickstart cannot find data

Reinstall the package and verify bundled files exist under `isotomics/input_data/`.
