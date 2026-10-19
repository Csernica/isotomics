from importlib import resources

import pandas as pd
import pytest

from isotomics import readCSVAndSimulate as sim

ALANINE = str(resources.files("isotomics") / "input_data" / "Example_Molecule_Input.csv")
SITES = ["Calphabeta", "Ccarboxyl", "Ocarboxyl", "Namine", "Hretained", "Hlost"]
ELEMENTS = ["C", "C", "O", "N", "H", "H"]
NUMBERS = [2, 1, 2, 1, 6, 2]

# Two hypothetical pathways forming fragment 44: one keeps Calphabeta, the other Ccarboxyl
PATHWAY_1 = [1, 0, 0, 1, 1, 0]
PATHWAY_2 = [0, 1, 0, 1, 1, 0]


def write_molecule_csv(path, fragments, relContributions=None):
    """fragments: {column name: site vector}. relContributions: {column name: share}, or None for no row."""
    df = pd.DataFrame({"Site Names": SITES, "Element": ELEMENTS, "Number Atoms": NUMBERS,
                       "Fragment full": [1] * len(SITES), **fragments})
    if relContributions is not None:
        df.loc[len(df)] = {"Site Names": "Relative Contribution", **relContributions}
    df.to_csv(path, index=False)
    return str(path)


def test_single_pathway_example_unchanged():
    molecule = sim.moleculeFromCsv(ALANINE, deltas=[0] * 6)
    assert molecule["fragmentationDictionary"] == {
        "full": {"01": {"subgeometry": [1, 1, 1, 1, 1, 1], "relCont": 1}},
        "44": {"01": {"subgeometry": [1, "x", "x", 1, 1, "x"], "relCont": 1}},
    }
    assert molecule["fragSubgeometryKeys"] == ["full_01", "44_01"]
    assert list(molecule["molecularDataFrame"].columns) == ["IDS", "Number", "Fragment full", "Fragment 44", "deltas"]


def test_subgeometries_read_from_csv(tmp_path):
    path = write_molecule_csv(tmp_path / "m.csv", {"Fragment 44_01": PATHWAY_1, "Fragment 44_02": PATHWAY_2},
                              {"Fragment 44_01": 0.4, "Fragment 44_02": 0.6})
    molecule = sim.moleculeFromCsv(path, deltas=[0] * 6)

    assert molecule["fragmentationDictionary"]["44"] == {
        "01": {"subgeometry": [1, "x", "x", 1, 1, "x"], "relCont": 0.4},
        "02": {"subgeometry": ["x", 1, "x", 1, 1, "x"], "relCont": 0.6},
    }
    assert molecule["fragmentationDictionary"]["full"]["01"]["relCont"] == 1
    assert molecule["fragSubgeometryKeys"] == ["full_01", "44_01", "44_02"]
    df = molecule["molecularDataFrame"]
    assert list(df.index) == SITES
    assert df["Number"].tolist() == NUMBERS


def test_mixed_fragment_is_weighted_sum_of_pathways(tmp_path):
    deltas = [-20, 15, 5, -3, 40, -60]

    def m1_fragment_44(fragments, relContributions=None):
        path = write_molecule_csv(tmp_path / f"m{len(list(tmp_path.iterdir()))}.csv", fragments, relContributions)
        predicted, _, _ = sim.simulateMeasurement(sim.moleculeFromCsv(path, deltas=deltas))
        return {sub: v["Abs. Abundance"] for sub, v in predicted["M1"]["44"].items()}

    one = m1_fragment_44({"Fragment 44": PATHWAY_1})
    two = m1_fragment_44({"Fragment 44": PATHWAY_2})
    mixed = m1_fragment_44({"Fragment 44_01": PATHWAY_1, "Fragment 44_02": PATHWAY_2},
                           {"Fragment 44_01": 0.4, "Fragment 44_02": 0.6})

    for sub in set(one) | set(two):
        assert mixed[sub] == pytest.approx(0.4 * one.get(sub, 0) + 0.6 * two.get(sub, 0), rel=1e-12)


def test_solver_recovers_sites_through_mixed_fragment(tmp_path):
    path = write_molecule_csv(tmp_path / "m.csv", {"Fragment 44_01": PATHWAY_1, "Fragment 44_02": PATHWAY_2},
                              {"Fragment 44_01": 0.4, "Fragment 44_02": 0.6})
    std = [-12.9, -10.95, 0, 0, 0, 0]
    smp = [-12.656, 21.685, 0, 0, 0, 0]
    output = sim.simulateSmpStd(path, std, smp, std, UValueList=["13C"], MNError=0.0001, UValueError=0.0001,  # fractions: 0.0001 = 0.1 per mil
                                MonteCarloN=20, plot=False, resultsFileName=str(tmp_path / "out.csv"))

    true_relative = [((1000 + s) / (1000 + t) - 1) * 1000 for s, t in zip(smp, std)]
    for site, truth in zip(SITES[:2], true_relative[:2]):
        assert output.loc[site, "Relative Deltas"] == pytest.approx(truth, abs=1)


def test_missing_relative_contributions_is_an_error(tmp_path):
    path = write_molecule_csv(tmp_path / "m.csv", {"Fragment 44_01": PATHWAY_1, "Fragment 44_02": PATHWAY_2})
    with pytest.raises(ValueError, match="Relative Contribution"):
        sim.moleculeFromCsv(path, deltas=[0] * 6)


def test_full_spectrum_includes_unsub(monkeypatch):
    # predictedMeasurement['U Values'] omits the unsubstituted molecule (U = 1); the spectrum must add it
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from isotomics import calcIsotopologues as ci
    from isotomics import spectrumVis

    molecule = sim.moleculeFromCsv(ALANINE, deltas=[-11.9, -9.95, 0, 0, 0, 0])
    predicted, _, _ = sim.simulateMeasurement(molecule, massThreshold=5)
    plotted = {}
    real_vlines = Axes.vlines
    monkeypatch.setattr(Axes, "vlines", lambda self, x, ymin, ymax, *a, **k: plotted.update(x=list(x), y=list(ymax)) or real_vlines(self, x, ymin, ymax, *a, **k))
    monkeypatch.setattr(spectrumVis.plt, "show", lambda *a, **k: None)

    spectrumVis.fullSpectrumVis(predicted["U Values"], molecule["molecularDataFrame"])
    labels = [t.get_text().split("\n")[-1] for t in spectrumVis.plt.gca().get_xticklabels()]
    relative = dict(zip(labels, plotted["y"]))

    bySub = ci.subDictionaryFromAtom(ci.inputToAtomDict(molecule["molecularDataFrame"], disable=True, M1Only=False))
    total = sum(v["Conc"] for v in bySub.values() if v["Mass"] <= 5)
    assert sum(plotted["y"]) == pytest.approx(1)
    assert relative["Unsub"] == pytest.approx(bySub[""]["Conc"] / total)
    assert relative["13C"] == pytest.approx(bySub["13C"]["Conc"] / total)


def test_spectrum_logY(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from isotomics import spectrumVis

    molecule = sim.moleculeFromCsv(ALANINE, deltas=[-11.9, -9.95, 0, 0, 0, 0])
    predicted, MNDict, _ = sim.simulateMeasurement(molecule, massThreshold=2)
    scales = []
    monkeypatch.setattr(spectrumVis.plt, "show", lambda *a, **k: scales.append(spectrumVis.plt.gca().get_yscale()))

    spectrumVis.fullSpectrumVis(predicted["U Values"], molecule["molecularDataFrame"], logY=True)
    spectrumVis.MNSpectrumVis(molecule["molecularDataFrame"], "44", predicted, "M1", MNDict, logY=True)
    spectrumVis.fullSpectrumVis(predicted["U Values"], molecule["molecularDataFrame"])
    assert scales == ["log", "log", "linear"]


UNRESOLVED_17O = {"M1": {"full": {"17O": "13C"}}}


def test_unresolved_relabels_isotopologues_documented_format():
    molecule = sim.moleculeFromCsv(ALANINE, deltas=[-11.9, -9.95, 0, 0, 0, 0])
    _, MNDict, _ = sim.simulateMeasurement(molecule, unresolvedDict=UNRESOLVED_17O)
    oxygen17 = [iso for iso in MNDict["M1"].values() if iso["Subs"] == "17O"]
    assert oxygen17 and all(iso["full_01 Subs"] == "13C" for iso in oxygen17)  # merged in the full molecule
    assert all(iso["44_01 Subs"] == "Unsub" for iso in oxygen17)  # fragment 44 loses the carboxyl O; unaffected


def test_unresolved_spectrum_columns():
    molecule = sim.moleculeFromCsv(ALANINE, deltas=[-11.9, -9.95, 0, 0, 0, 0])
    plain, _, _ = sim.simulateMeasurement(molecule)
    merged, _, _ = sim.simulateMeasurement(molecule, unresolvedDict=UNRESOLVED_17O)
    full = plain["M1"]["full"]
    assert "17O" not in merged["M1"]["full"]
    assert merged["M1"]["full"]["13C"]["Rel. Abundance"] == pytest.approx(full["13C"]["Rel. Abundance"])
    assert merged["M1"]["full"]["13C"]["Combined Rel. Abundance"] == pytest.approx(
        full["13C"]["Rel. Abundance"] + full["17O"]["Rel. Abundance"])
    assert merged["M1"]["44"] == plain["M1"]["44"]


def test_simulateSmpStd_unresolved_matches_omitted(tmp_path):
    std = [-12.9, -10.95, 0, 0, 0, 0]
    smp = [-12.656, 21.685, 0, 0, 0, 0]
    common = dict(UValueList=["13C"], MNError=1e-6, UValueError=1e-6, MonteCarloN=5, plot=False,
                  resultsFileName=str(tmp_path / "out.csv"))
    omitted = sim.simulateSmpStd(ALANINE, std, smp, std, omitMeasurements={"M1": {"full": ["17O"]}}, **common)
    unresolved = sim.simulateSmpStd(ALANINE, std, smp, std, unresolvedDict=UNRESOLVED_17O, **common)
    for site in ["Calphabeta", "Ccarboxyl", "Namine"]:
        assert unresolved.loc[site, "Relative Deltas"] == pytest.approx(omitted.loc[site, "Relative Deltas"], abs=0.2)
