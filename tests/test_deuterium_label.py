"""Deuterium is labelled '2H' throughout; older data labelled 'D' is renamed on read, with a warning."""
from importlib import resources
import warnings

import pandas as pd
import pytest

from isotomics import basicDeltaOperations as op
from isotomics import dataAnalyzerMNIsoX as dA
from isotomics import readCSVAndSimulate as sim
from isotomics import solveExperimentalData

DATA = resources.files("isotomics") / "input_data"
MOLECULE = str(DATA / "Example_Molecule_Input.csv")
STD = [-11.9, -9.95, 0, 0, 0, 0]


@pytest.fixture(autouse=True)
def reset_rename_warnings():
    # The rename warning is shown once per source per session; reset so each test sees it
    op._warnedLegacySources.clear()


@pytest.mark.parametrize("old, new", [
    ("D", "2H"), ("D-13C", "2H-13C"), ("13C-D-D", "13C-2H-2H"), ("D/Unsub", "2H/Unsub"),
    ("2H", "2H"), ("13C", "13C"), ("Unsub", "Unsub"), ("Fragment 44", "Fragment 44"),
])
def test_standardize_substitution_name(old, new):
    assert op.standardizeSubstitutionName(old) == new


def test_simulation_uses_2H():
    molecule = sim.moleculeFromCsv(MOLECULE, deltas=STD)
    predicted, _, _ = sim.simulateMeasurement(molecule)
    peaks = set(predicted["M1"]["44"]) | set(predicted["M1"]["full"]) | set(predicted["U Values"])
    assert "2H" in peaks
    assert not any(op.standardizeSubstitutionName(p) != p for p in peaks)


def test_isox_with_D_is_read_as_2H():
    # The bundled raw .isox files label deuterium 'D'
    isox = DATA / "Experimental_Data" / "44" / "Smp" / "20211006_07_Alanine_C1-1_M1_rep1.isox"
    with pytest.warns(UserWarning, match="'D' to '2H'"):
        df = dA.readIsoX(str(isox))
    assert "2H" in set(df["isotopolog"])
    assert "D" not in set(df["isotopolog"])


def test_error_file_with_D_matches_2H(tmp_path):
    canonical = (DATA / "Alanine Experimental Errors.csv").read_text(encoding="utf-8-sig")
    legacy = tmp_path / "legacy_errors.csv"
    legacy.write_text(canonical.replace("2H,", "D,"), encoding="utf-8")

    with pytest.warns(UserWarning, match="'D' to '2H'"):
        legacyErrors = sim.extractExperimentalErrors(str(legacy))
    assert legacyErrors == sim.extractExperimentalErrors(str(DATA / "Alanine Experimental Errors.csv"))


def test_processed_data_with_D_gives_same_result():
    canonical = pd.read_csv(DATA / "Experimental_Data_Input.csv")
    legacy = canonical.copy()
    legacy["MN Relative Abundance"] = legacy["MN Relative Abundance"].replace("2H", "D")
    legacy["IsotopeRatio"] = legacy["IsotopeRatio"].replace("2H/Unsub", "D/Unsub")

    args = dict(cwd=".", MOLECULE_INPUT_PATH=MOLECULE, std_deltas=STD, mAObs=-0.8, mARelErr=0.0001,
                perturbTheoryOAmt=0.0003, MonteCarloN=20, plot=False, resultsFileName=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # canonical input must not warn
        expected = solveExperimentalData.experimentalDataM1(canonical, **args)
    with pytest.warns(UserWarning, match="'D' to '2H'"):
        result = solveExperimentalData.experimentalDataM1(legacy, **args)
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_argument_nested_and_not_mutated():
    arg = {"M1": {"44": ["D", "13C"], "full": ("D-13C",)}, "U": {"D/Unsub": 0.1}}
    with pytest.warns(UserWarning, match="the test argument"):
        out = op.standardizeSubstitutionArgument(arg, "the test argument")
    assert out == {"M1": {"44": ["2H", "13C"], "full": ("2H-13C",)}, "U": {"2H/Unsub": 0.1}}
    assert arg["M1"]["44"] == ["D", "13C"]  # input untouched
    assert op.standardizeSubstitutionArgument(None, "x") is None
    assert op.standardizeSubstitutionArgument(0.001, "x") == 0.001


def test_omitMeasurements_with_D_matches_2H():
    molecule = sim.moleculeFromCsv(MOLECULE, deltas=STD)
    expected, _, _ = sim.simulateMeasurement(molecule, omitMeasurements={"M1": {"44": ["2H"]}})
    with pytest.warns(UserWarning, match="omitMeasurements"):
        legacy, _, _ = sim.simulateMeasurement(molecule, omitMeasurements={"M1": {"44": ["D"]}})
    assert "2H" not in legacy["M1"]["44"]
    assert legacy["M1"] == expected["M1"]


def test_UValue_with_D_matches_2H():
    molecule = sim.moleculeFromCsv(MOLECULE, deltas=STD)
    expected = solveExperimentalData.getUVal(None, molecule, UValue="2H/Unsub", mAObs=-50, mARelErr=0.001)
    with pytest.warns(UserWarning, match="UValue"):
        legacy = solveExperimentalData.getUVal(None, molecule, UValue="D/Unsub", mAObs=-50, mARelErr=0.001)
    assert legacy == expected and "2H" in legacy


def test_targetRatio_with_D_matches_2H(monkeypatch):
    from isotomics import dataScreenIsoX
    import matplotlib
    matplotlib.use("Agg")
    isox = str(DATA / "Experimental_Data" / "44" / "Smp" / "20211006_07_Alanine_C1-1_M1_rep1.isox")
    _, merged = dA.calc_Folder_Output([isox], MNRelativeAbundance=True)
    plotted = []
    monkeypatch.setattr(dataScreenIsoX.plt, "show", lambda *a, **k: None)
    real_plot = dataScreenIsoX.plt.Axes.plot
    monkeypatch.setattr(dataScreenIsoX.plt.Axes, "plot", lambda self, *a, **k: plotted.append(list(a[-1])) or real_plot(self, *a, **k))

    dataScreenIsoX.visualizeTICVersusTime(isox, merged, targetRatio="2H")
    with pytest.warns(UserWarning, match="targetRatio"):
        dataScreenIsoX.visualizeTICVersusTime(isox, merged, targetRatio="D")
    assert len(plotted) == 2 and plotted[0] == plotted[1]
