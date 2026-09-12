"""Bounded physical search controls, with serialized local field/navigation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly_arena.config import load_config
from fly_arena.ethology import EthologySimulation
from scripts.check_ethology_scenarios import verify_resume, execute_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/search-candidate-v5.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/search-acceptance")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["initial"]["dust_by_region"] = dict.fromkeys(cfg["initial"]["dust_by_region"], 0.)
    for patch in cfg["food"]:
        patch["x"] += 12
        patch["y"] += 12
    sim = EthologySimulation(cfg, seed=101, brain_enabled=False)
    try:
        checkpoint_at = None
        for _ in range(400):
            sim.step()
            if sim.last_decision.action == "WALK" and sim.navigator.phase == "CAST" and sim.navigator.leg_elapsed > .05:
                checkpoint_at = sim.organism.clocks.physics_time_s
                break
        assert checkpoint_at is not None
    finally:
        sim.close()
    results = {"cast_resume": verify_resume("search-cast", cfg, checkpoint_at, args.output)}

    odor_only = load_config(args.config)
    odor_only["initial"]["dust_by_region"] = dict.fromkeys(odor_only["initial"]["dust_by_region"], 0.)
    odor_only["food"] = [dict(id="odor", x=3., y=0., radius=1., amount=0.,
                             energy_density=0., taste=0., odor=1., odor_when_empty=True)]
    result = execute_case("odor-only", odor_only, 3., seed=101, output=args.output)
    assert result["final"]["ingested_total"] == 0
    assert result["peak_energy"] <= result["initial_energy"]
    assert result["action_durations"].get("FEED", 0) == 0
    results["odor_only"] = result

    no_wind = load_config(args.config)
    no_wind["environment"].setdefault("plume", {})["wind_speed_mm_s"] = 0.
    no_wind["initial"]["dust_by_region"] = dict.fromkeys(no_wind["initial"]["dust_by_region"], 0.)
    no_wind["food"] = []
    sim = EthologySimulation(no_wind, seed=101, brain_enabled=False)
    try:
        for _ in range(100):
            sim.step()
        assert sim.navigator.phase == "NO_WIND"
        assert sim.minimum_upright > .8
        results["no_wind"] = sim.telemetry()
    finally:
        sim.close()

    refill = load_config(scenario="feeding-contact")
    refill["food"][0]["amount"] = .1
    refill["environment"]["recurring_events"] = [dict(kind="refill", time_s=1.1, repeat_s=1.,
        food_id=refill["food"][0]["id"], amount=.25, capacity=.25)]
    sim = EthologySimulation(refill, seed=1, brain_enabled=False)
    try:
        refills = []
        empty_seen = False
        for _ in range(160):
            step = sim.step()
            empty_seen |= sim.environment.food[0].amount == 0
            refills += [e for e in step["external_events"] if e["kind"] == "refill"]
        assert empty_seen and len(refills) == 1
        assert sim.organism.state.ingested_total > .1
        assert abs(sim.organism.state.ingested_total - .35) < 1e-8
        results["refill"] = {"events": refills, "telemetry": sim.telemetry()}
    finally:
        sim.close()
    results["status"] = "pass"
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SEARCH PHYSICAL CONTROLS: PASS", flush=True)


if __name__ == "__main__":
    main()
