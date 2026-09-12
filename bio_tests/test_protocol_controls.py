"""End-to-end controls for the display -> retina -> neural-report boundary.

The tiny graph contains an ON branch, an OFF branch and their joint readout.
These are numerical protocol checks, not biological validation of a circuit.
"""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from fly_bio.benchmark import run_case


def _toy_graph():
    types = np.array(["retina", "retina", "L1", "L2", "Mi1", "Tm1",
                      "T4a", "T5a", "LC4", "LPLC2", "DNp01"])
    matrix = csr_matrix(([-1., -1., -1., 1., 1., 1., 1., 1., .8, .2],
                         ([2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
                        shape=(11, 11), dtype=np.float32)

    def indices(names):
        names = [names] if isinstance(names, str) else names
        return np.flatnonzero(np.isin(types, names)).astype(np.int32)

    return SimpleNamespace(
        n=11, edge_count=matrix.nnz, incoming=matrix, indices=indices, types=types,
        ports={"retina": [0, 1], "eye": [0, 1],
               "uv": [[.65, .5], [.65, .5]]},
    )


def _trace(report, output):
    with np.load(output / report["trace"]) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_distinct_display_contrasts_keep_distinct_result_artifacts(tmp_path):
    weak = run_case(_toy_graph(), tmp_path, kind="dark_loom", contrast=.2)
    strong = run_case(_toy_graph(), tmp_path, kind="dark_loom", contrast=.8)
    assert weak["trace"] != strong["trace"]
    assert (tmp_path / weak["trace"]).exists() and (tmp_path / strong["trace"]).exists()


def test_shrinking_has_no_artificial_onset_response_before_motion(tmp_path):
    """A pre-existing large disk must not act like a newly appearing flash."""
    report = run_case(_toy_graph(), tmp_path, kind="shrinking")
    trace = _trace(report, tmp_path)
    before_motion = trace["times_s"] <= report["display"]["pre_s"]
    assert np.all(trace["retinal_luminance"][before_motion] < .11)
    for name in [key for key in trace if key.endswith("_maximum_abs")]:
        np.testing.assert_array_equal(trace[name][before_motion], 0.)
    assert report["populations"]["GF"]["peak_rms_delta_au"] > .001


def test_freezing_retina_removes_visual_modulation_despite_changing_movie(tmp_path):
    """A changing stimulus cannot enter the graph downstream of frozen retina."""
    active = run_case(_toy_graph(), tmp_path, kind="dark_loom")
    blocked = run_case(_toy_graph(), tmp_path, kind="dark_loom", intervention="retina")
    active_trace, blocked_trace = _trace(active, tmp_path), _trace(blocked, tmp_path)
    np.testing.assert_array_equal(active_trace["retinal_luminance"],
                                  blocked_trace["retinal_luminance"])
    assert np.ptp(blocked_trace["retinal_luminance"]) > .3
    assert active["populations"]["GF"]["peak_rms_delta_au"] > .001
    assert blocked["frozen_cells"] == 2
    for population in blocked["populations"].values():
        assert population["peak_rms_delta_au"] == 0.


def test_gf_clamp_removes_readout_modulation_without_erasing_upstream_response(tmp_path):
    """A clamped endpoint is a readout intervention, not removal of the stimulus."""
    report = run_case(_toy_graph(), tmp_path, kind="dark_loom", intervention="gf")
    assert report["populations"]["GF"]["peak_rms_delta_au"] == 0.
    assert report["populations"]["LC4"]["peak_rms_delta_au"] > .001
    assert report["populations"]["LPLC2"]["peak_rms_delta_au"] > .001
    assert report["frozen_cells"] == 1
