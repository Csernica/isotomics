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


def test_readIsoX_rejects_multiple_compounds(tmp_path):
    isox = resources.files("isotomics") / "input_data" / "Experimental_Data" / "44" / "Smp" / "20211006_07_Alanine_C1-1_M1_rep1.isox"
    one = pd.read_csv(isox, sep="\t")
    dA.readIsoX(str(isox))  # a single compound reads normally

    other = one.copy()
    other["compound"] = "X"
    path = tmp_path / "two_fragments.isox"
    pd.concat([one, other]).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="this file contains compounds Ala44 and X; isotomics expects one fragment per file"):
        dA.readIsoX(str(path))


def test_scans_kept_when_first_peak_is_missing(tmp_path):
    # '13C' sorts first among the peaks; remove it from 200 scans inside the time window
    isox = resources.files("isotomics") / "input_data" / "Experimental_Data" / "44" / "Smp" / "20211006_07_Alanine_C1-1_M1_rep1.isox"
    raw = pd.read_csv(isox, sep="\t")
    inWindow = raw.loc[raw["time.min"].between(5, 15), "scan.no"].drop_duplicates()
    dropScans = set(inWindow.iloc[1000:1200])
    thinned = raw[~((raw["isotopolog"] == "13C") & raw["scan.no"].isin(dropScans))]
    path = tmp_path / "missing13C.isox"
    thinned.to_csv(path, sep="\t", index=False)

    _, merged = dA.calc_Folder_Output([str(path)], timeBounds=(5, 15), MNRelativeAbundance=True,
                                      RSESNScreen=False, zeroCountsScreen=False, peakDriftScreen=False)
    df = next(iter(merged.values()))["mergedDf"]
    assert "isotopolog" not in df.columns
    assert set(inWindow) <= set(df["scanNumber"])                       # no scans lost
    missing = df[df["scanNumber"].isin(dropScans)]
    assert len(missing) == 200 and (missing["counts13C"] == 0).all()   # 13C absent there...
    assert (missing["retTime"] > 0).all() and (missing["tic"] > 0).all()  # ...but the scans keep their time and TIC


def test_MN_relative_abundance_shot_noise_matches_counting_statistics():
    # With pure Poisson counting noise, the observed RSE of each M+N relative abundance should equal its
    # shot-noise limit (eq. S.12), so RSE/SN = 1 for every peak, including the dominant one
    rng = np.random.default_rng(0)
    nScans = 20000
    mergedDf = pd.DataFrame({"counts" + k: rng.poisson(v, nScans).astype(float)
                             for k, v in {"13C": 4000, "Unsub": 1130, "15N": 650, "2H": 155}.items()})
    subs = ["13C", "15N", "2H", "Unsub"]
    mergedDf["total Counts"] = mergedDf[["counts" + s for s in subs]].sum(axis=1)
    for s in subs:
        mergedDf["MN Relative Abundance " + s] = mergedDf["counts" + s] / mergedDf["total Counts"]
    mergedDf["tic"] = 1.0

    out = dA.output_Raw_File_MN_Rel_Abundance(mergedDf, subs, massStr="44")["44"]
    for s in subs:
        assert out[s]["RelStError"] / out[s]["ShotNoiseLimit"] == pytest.approx(1, abs=0.05), s
