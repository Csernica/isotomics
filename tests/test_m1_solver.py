from pathlib import Path

import numpy as np

from isotomics import dataAnalyzerMNIsoX as data_analyzer
from isotomics import organizeData
from isotomics import solveExperimentalData


def test_m1_solver_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    cwd = repo_root
    molecule_input_path = "Input Data/Example_Molecule_Input.csv"
    standard_delta_approx = [-11.9, -9.95, 0, 0, 0, 0]

    fragment_folder_paths = organizeData.get_subfolder_paths(
        str(repo_root / "Input Data" / "Experimental_Data")
    )
    rtn_means, _ = data_analyzer.processIndividualAndAverageIsotopeRatios(
        fragment_folder_paths,
        cwd,
        outputToCSV=True,
        csvOutputPath="all_data_output.csv",
        file_extension=(".isox",),
        processed_data_subfolder="Input Data",
        time_bounds=(5, 15),
        peakDriftScreen=True,
        peakDriftThreshold=2,
        zeroCountsScreen=True,
        zeroCountsThreshold=0,
        RSESNScreen=True,
    )

    output_df = solveExperimentalData.experimentalDataM1(
        rtn_means,
        cwd,
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
    assert (repo_root / "M1Output.csv").exists()
    assert (repo_root / "Input Data" / "all_data_output.csv").exists()
