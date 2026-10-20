from importlib import resources

import numpy as np
import pandas as pd
import pytest

from isotomics import dataAnalyzerMNIsoX as dA
from isotomics import dataScreenIsoX as screen


def rse_table():
    return pd.DataFrame({
        "FileName": ["a.isox", "b.isox"],
        "Fragment": ["44", "44"],
        "IsotopeRatio": ["13C/Unsub", "15N/Unsub"],
        "RelStdError": [1.5e-4, 3.0e-4],
        "ShotNoise": [1.0e-4, 1.0e-4],  # RSE/SN of 1.5 and 3
    })


@pytest.mark.parametrize("threshold, flagged", [(None, ["15N/Unsub"]), (1, ["13C/Unsub", "15N/Unsub"]), (5, [])])
def test_RSESNScreen_uses_threshold(capsys, threshold, flagged):
    kwargs = {} if threshold is None else {"threshold": threshold}  # None: the default, 2
    screen.RSESNScreen(rse_table(), **kwargs)
    out = capsys.readouterr().out
    for ratio in ["13C/Unsub", "15N/Unsub"]:
        assert (ratio in out) == (ratio in flagged)


def test_peakDriftScreen_names_the_drifted_peak(capsys):
    unsub = 44.05002
    drift = 5e-6 * (unsub + screen.getThisSubMass("13C"))  # 5 ppm, on the 13C peak only
    n = 50
    mergedDf = pd.DataFrame({
        "countsUnsub": np.full(n, 1000.0), "counts13C": np.full(n, 600.0), "counts15N": np.full(n, 100.0),
        "massUnsub": np.full(n, unsub),
        "mass13C": np.full(n, unsub + screen.getThisSubMass("13C") + drift),
        "mass15N": np.full(n, unsub + screen.getThisSubMass("15N")),
    })
    screen.peakDriftScreen({"file.isox": {"subNameList": ["Unsub", "13C", "15N"], "mergedDf": mergedDf}}, threshold=2)
    out = capsys.readouterr().out
    assert "file.isox 13C" in out and "15N" not in out
    ppm = float(out.strip().split("with size ")[-1])
    assert ppm == pytest.approx(5, rel=1e-3)


def test_RSESNThreshold_reaches_the_screen(capsys):
    isox = str(resources.files("isotomics") / "input_data" / "Experimental_Data" / "44" / "Smp" /
               "20211006_07_Alanine_C1-1_M1_rep1.isox")
    common = dict(timeBounds=(5, 15), MNRelativeAbundance=True, zeroCountsScreen=False, peakDriftScreen=False)
    dA.calc_Folder_Output([isox], **common)
    assert "fails RSE/SN test" not in capsys.readouterr().out
    dA.calc_Folder_Output([isox], RSESNThreshold=0, **common)  # every measurement exceeds 0
    assert "fails RSE/SN test" in capsys.readouterr().out
