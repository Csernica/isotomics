from importlib import resources

import numpy as np
import pandas as pd

from isotomics import dataAnalyzerMNIsoX as data_analyzer
from isotomics import organizeData
from isotomics import solveExperimentalData


def test_m1_solver_smoke(tmp_path, monkeypatch):
    # Read the example data bundled with the package; write all outputs to a temp dir.
    monkeypatch.chdir(tmp_path)
    input_root = resources.files("isotomics").joinpath("input_data")
    molecule_input_path = str(input_root / "Example_Molecule_Input.csv")
    standard_delta_approx = [-11.9, -9.95, 0, 0, 0, 0]

    fragment_folder_paths = organizeData.get_subfolder_paths(
        str(input_root / "Experimental_Data")
    )
    rtn_means, _ = data_analyzer.processIndividualAndAverageIsotopeRatios(
        fragment_folder_paths,
        tmp_path,
        outputToCSV=True,
        csvOutputPath="all_data_output.csv",
        file_extension=(".isox",),
        processed_data_subfolder=".",
        time_bounds=(5, 15),
        peakDriftScreen=True,
        peakDriftThreshold=2,
        zeroCountsScreen=True,
        zeroCountsThreshold=0,
        RSESNScreen=True,
    )

    output_df = solveExperimentalData.experimentalDataM1(
        rtn_means,
        tmp_path,
        molecule_input_path,
        standard_delta_approx,
        UValue="13C/Unsub",
        mAObs=-0.8,
        mARelErr=0.1,
        perturbTheoryOAmt=0.0003,
        MonteCarloN=50,
        resultsFileName="M1Output.csv",
        plot=False,
    )

    assert len(output_df) > 0
    assert "CRF Deltas" in output_df.columns
    assert "CRF Deltas Error" in output_df.columns
    assert "Relative Deltas" in output_df.columns
    assert "Relative Deltas Error" in output_df.columns
    assert "Standard Delta Values in CRF" in output_df.columns
    assert output_df["Standard Delta Values in CRF"].tolist() == standard_delta_approx
    assert np.isfinite(output_df["CRF Deltas"]).all()
    assert np.isfinite(output_df["CRF Deltas Error"]).all()
    assert (tmp_path / "M1Output.csv").exists()
    written = pd.read_csv(tmp_path / "M1Output.csv")
    assert written["Site Names"].tolist() == output_df.index.tolist()
    assert (tmp_path / "all_data_output.csv").exists()
