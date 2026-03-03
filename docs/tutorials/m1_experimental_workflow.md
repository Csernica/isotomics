# M+1 Experimental Workflow

This tutorial mirrors the core notebook workflow:

1. Build fragment folder paths from an `Experimental_Data` directory.
2. Process `.isox` files into mean sample/standard ratios.
3. Run `solveExperimentalData.experimentalDataM1` to recover site-specific values.

Core functions:

- `isotomics.dataAnalyzerMNIsoX.processIndividualAndAverageIsotopeRatios`
- `isotomics.solveExperimentalData.experimentalDataM1`
