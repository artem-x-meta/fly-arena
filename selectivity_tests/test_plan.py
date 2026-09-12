import json

import pytest

from fly_bio_selectivity import experiment


def test_preregistration_is_serializable_repeatable_and_refuses_silent_replacement(tmp_path, monkeypatch):
    plan = experiment.make_plan(tmp_path)
    before = (tmp_path / "plan.json").read_bytes()
    assert len(plan["trials"]) == 48
    assert json.loads(before)["candidate_delays_s"] == {"Mi1": .018, "Tm1": .013}
    experiment.make_plan(tmp_path)
    assert (tmp_path / "plan.json").read_bytes() == before
    monkeypatch.setattr(experiment, "sources", lambda: {"different.py": "changed"})
    with pytest.raises(ValueError, match="incompatible plan"):
        experiment.make_plan(tmp_path)
    assert (tmp_path / "plan.json").read_bytes() == before
