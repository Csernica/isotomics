import os
from importlib import resources
from pathlib import Path

from . import readCSVAndSimulate as sim
from . import organizeData
from . import dataAnalyzerMNIsoX as dA
from . import solveExperimentalData


# Experimental quickstart (based on the alanine example)


def main():
    cwd = Path().resolve()
    packaged_input_root = resources.files("isotomics").joinpath("input_data")

    with resources.as_file(packaged_input_root) as quickstart_input:
        # STEP 1: molecule definition from CSV
        MOLECULE_INPUT_PATH = quickstart_input / "Example_Molecule_Input.csv"

        # STEP 3: experimental processing parameters
        ACQUISITION_LENGTH = (5, 15)
        FILE_EXTENSION = ".isox"

        OUTPUT_TO_CSV = True
        CSV_OUTPUT_PATH = "all_data_output.csv"

        PEAK_DRIFT_SCREEN = True
        PEAK_DRIFT_THRESHOLD = 2
        ZERO_COUNTS_SCREEN = True
        ZERO_COUNTS_THRESHOLD = 0
        RSE_SN_SCREEN = True

        # Molecular-average constraints (from external measurement or Orbitrap)
        # Standard: Calphabeta = mean of C-2 (-12.9) and C-3 (-7.0), Ccarboxyl = C-1 (-11.9); Weiss et al. (2023), Table 2
        STANDARD_DELTA_APPROX = [-9.95, -11.9, 0, 0, 0, 0]
        MOLECULAR_AVERAGE_SAMPLE_COMPOSITION = -0.8
        MOLECULAR_AVG_SAMPLE_ERROR = 0.0001  # relative error as a fraction (0.1 per mil)
        U_VALUE = "13C/Unsub"

        # Initialize molecule
        sim.moleculeFromCsv(str(MOLECULE_INPUT_PATH), deltas=STANDARD_DELTA_APPROX)

        # STEP 4: process data from packaged example folders
        new_folder_path = quickstart_input / "Experimental_Data"
        fragmentFolderPaths = organizeData.get_subfolder_paths(str(new_folder_path))
        rtnMeans, _ = dA.processIndividualAndAverageIsotopeRatios(
            fragmentFolderPaths,
            cwd,
            outputToCSV=OUTPUT_TO_CSV,
            csvOutputPath=CSV_OUTPUT_PATH,
            file_extension=FILE_EXTENSION,
            processed_data_subfolder=".",
            time_bounds=ACQUISITION_LENGTH,
            peakDriftScreen=PEAK_DRIFT_SCREEN,
            peakDriftThreshold=PEAK_DRIFT_THRESHOLD,
            zeroCountsScreen=ZERO_COUNTS_SCREEN,
            zeroCountsThreshold=ZERO_COUNTS_THRESHOLD,
            RSESNScreen=RSE_SN_SCREEN,
        )

        # STEP 5: Skip, data screening (run in notebook)

        # STEP 6: solve for site-specific values
        solveExperimentalData.experimentalDataM1(
            rtnMeans,
            cwd,
            str(MOLECULE_INPUT_PATH),
            STANDARD_DELTA_APPROX,
            UValue=U_VALUE,
            mAObs=MOLECULAR_AVERAGE_SAMPLE_COMPOSITION,
            mARelErr=MOLECULAR_AVG_SAMPLE_ERROR,
            perturbTheoryOAmt=0.0003,
            plotPath="M1Output.png",
        )


if __name__ == "__main__":
    main()

