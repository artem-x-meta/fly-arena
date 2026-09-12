"""Two organisms sharing a patch, with optional engineered resource defence.

Each fly gets its own organism, behaviour arbiter, navigator and motor owner,
all taken unchanged from the single-fly package. The environment is shared, so
the food both of them eat comes out of the same finite amount. That, plus one
declared contact pair between their bodies, is the whole of the competition.

The passive mode has no social requests. The optional policy adds local rival
perception, orientation, body pushing and retreat through existing walking.
It is not a neural reconstruction or a model of biological lunges.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from fly_arena.behavior import BehaviorController
from fly_arena.config import load_config
from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_arena.search import SurgeCastNavigator
from fly_arena.sensors import NeuralReadout

from .arena import DuelArena
from .social import SocialController
from .resources import apply_shared_events


class Contestant:
    """One fly's organism, decision maker and navigator."""

    def __init__(self, unit, config, seed):
        self.unit = unit
        self.organism = Organism(config.get("organism", {}), initial=config.get("initial", {}))
        self.behavior = BehaviorController(config.get("behavior", {}), seed=seed)
        self.navigator = SurgeCastNavigator(config["navigation"], seed)
        self.frame = None
        self.decision = None
        self.action_durations: dict[str, float] = {}
        self.patch_time_s = 0.0
        self.centre_time_s = 0.0
        self.contact_time_s = 0.0
        self.odor_samples: list[float] = []
        self.social = SocialController(seed=seed)
        self.social_frame = None
        self.social_decision = None
        self.min_upright = 1.0
        self.contact_push_s = 0.
        self.fence_contact_s = 0.
        self.first_feed_s = None
        self.food_after_retreat = 0.

    @property
    def name(self):
        return self.unit.name


class Duel:
    """One shared world, two contestants, one 10 ms exchange for both."""

    def __init__(self, config, *, seed=1, body_contact=True, start=None, names=("fly", "rival"),
                 start_headings=None, social=False, social_sensing=True):
        self.config = config
        self.seed = seed
        self.social_enabled = bool(social)
        self.social_sensing = bool(social_sensing)
        self.environment = Environment(config["food"], config.get("environment", {}),
                                       config.get("events", ()), seed=seed)
        patches = [{"id": patch.id, "x": patch.x, "y": patch.y, "radius": patch.radius,
                    "surface_z": getattr(patch, "surface_z", .01)} for patch in self.environment.food]
        self.arena = DuelArena(seed=seed, food_patches=patches, body_contact=body_contact,
                               start=start, names=names, start_headings=start_headings,
                               head_contact=config.get("duel", {}).get("head_contact", False),
                               collision_geometry=config.get("duel", {}).get("collision_geometry", "mesh"))
        self.contestants = [Contestant(unit, config, seed + index)
                            for index, unit in enumerate(self.arena.units)]
        self.environment.initial_dust = sum(sum(c.organism.state.dust_by_region.values())
                                            for c in self.contestants)
        for c in self.contestants:
            c.unit.motor.adaptive_proboscis = config.get("motor", {}).get("adaptive_proboscis", False)
        self.steps = 0
        self.body_contact_events = 0
        self.any_contact_s = 0.
        self.last_transfers = []

    @property
    def elapsed_s(self):
        return self.contestants[0].organism.clocks.physics_time_s

    def step(self, dt=.01):
        if not np.isfinite(dt) or abs(dt - .01) > 1e-12:
            raise ValueError("Duel exchange is fixed at 10 ms")
        time_s = self.elapsed_s
        external = self.environment.apply_scheduled_events(time_s, self.contestants[0].organism)
        social_frames = self.arena.social_frames(enabled=self.social_sensing)
        decisions, gaits = [], []
        for contestant, social_frame in zip(self.contestants, social_frames):
            unit = contestant.unit
            frame = self.environment.sense(unit, contestant.organism, None)
            decision = contestant.behavior.decide(frame, contestant.organism, NeuralReadout(),
                                                  dt=dt, mode="ethology-hybrid")
            contestant.social_frame = social_frame
            command = contestant.social.step(social_frame, frame, contestant.organism, dt=dt,
                enabled=self.social_enabled and self.social_sensing, base_action=decision.action)
            if not decision.gates.get("stable", True):
                command = None
            contestant.social_decision = command
            if command is not None:
                # The original arbiter remains the sole actuator owner. Keep
                # behavior timers synchronized with the actual motor action.
                decision = contestant.behavior._choose("WALK", command.reason, "engineered_social_policy",
                    decision.gates, decision.scores, time_s)
                if command.state == "FENCE":
                    decision = replace(decision, action="FENCE")
                elif command.state == "ASSESS":
                    decision = replace(decision, action="IDLE")
            walking = decision.action == "WALK" and command is None
            gait = contestant.navigator.step(frame, NeuralReadout(), "ethology-hybrid", dt, active=walking)
            if not walking:
                gait = np.zeros(2)
            if command is not None:
                gait = np.asarray(command.gait)
            contestant.frame, contestant.decision = frame, decision
            decisions.append(decision)
            gaits.append(gait)

        self.arena.track_grooming = any(sum(c.organism.state.dust_by_region.values()) > 0
                                        for c in self.contestants)
        events = self.arena.step(decisions, gaits)

        touching = self.arena.contacts_between_flies() > 0
        self.body_contact_events += int(touching)
        physical_social = self.arena.social_frames()
        self.any_contact_s += dt if any(s.body_contact for s in physical_social) else 0.
        self.last_transfers = apply_shared_events(self.environment, events,
                                                  [c.organism for c in self.contestants], dt)
        for index, contestant in enumerate(self.contestants):
            contestant.organism.advance(dt, contestant.decision.action,
                                        movement_proxy=max(0., contestant.frame.movement))
            ingested = sum(self.last_transfers[index]["intake_by_food"].values())
            if ingested > 0 and contestant.first_feed_s is None:
                contestant.first_feed_s = time_s + dt
            if contestant.social.counters["retreats"] > 0:
                contestant.food_after_retreat += ingested
            contestant.min_upright = min(contestant.min_upright, contestant.unit.upright)
            if (contestant.social_decision is not None and contestant.social_decision.state == "PUSH"
                    and physical_social[index].body_contact):
                contestant.contact_push_s += dt
            if contestant.social_decision is not None and contestant.social_decision.state == "FENCE":
                contestant.fence_contact_s += self.arena.foreleg_contact_s[index]
            name = contestant.decision.action
            contestant.action_durations[name] = contestant.action_durations.get(name, 0.) + dt
            contestant.contact_time_s += dt if touching else 0.
            odour = (contestant.frame.odor_left + contestant.frame.odor_right) / 2
            contestant.odor_samples.append(odour)
            # Distance to the nearest patch. "At the patch" is the radius plus a
            # body length, because a feeding fly stands at the rim with only its
            # proboscis inside; "holding" it is the radius itself.
            nearest = min(float(np.linalg.norm(contestant.unit.position[:2] - np.array([patch.x, patch.y])))
                          - patch.radius for patch in self.environment.food)
            contestant.patch_time_s += dt if nearest <= 1.2 else 0.
            contestant.centre_time_s += dt if nearest <= 0. else 0.
        self.environment.advance_field(dt)
        self.steps += 1

    def summary(self):
        remaining = sum(patch.amount for patch in self.environment.food)
        rows = []
        for contestant in self.contestants:
            state = contestant.organism.state
            rows.append({"name": contestant.name,
                         "ingested": round(state.ingested_total, 4),
                         "energy": round(state.energy, 2),
                         "patch_time_s": round(contestant.patch_time_s, 2),
                         "centre_time_s": round(contestant.centre_time_s, 2),
                         "actions": {k: round(v, 2) for k, v in
                                     sorted(contestant.action_durations.items(), key=lambda kv: -kv[1])},
                         "mean_odor": round(float(np.mean(contestant.odor_samples)), 4)
                         if contestant.odor_samples else 0.,
                         "upright_now": round(contestant.unit.upright, 3),
                         "min_upright": round(contestant.min_upright, 5),
                         "first_feed_s": contestant.first_feed_s,
                         "food_after_retreat": round(contestant.food_after_retreat, 4),
                         "contact_push_s": round(contestant.contact_push_s, 4),
                         "fence_contact_s": round(contestant.fence_contact_s, 6),
                         "social": contestant.social.summary()})
        eaten = sum(row["ingested"] for row in rows)
        share = max(row["ingested"] for row in rows) / eaten if eaten > 1e-9 else None
        return {"seconds": round(self.elapsed_s, 2), "food_remaining": round(remaining, 4),
                "eaten_total": round(eaten, 4),
                "winner_share": round(share, 3) if share is not None else None,
                "body_contact_steps": self.body_contact_events,
                "any_contact_exchange_s": round(self.any_contact_s, 4),
                "social_enabled": self.social_enabled, "social_sensing": self.social_sensing,
                "collision_geometry": self.arena.collision_geometry,
                "max_interfly_penetration_mm": self.arena.max_interfly_penetration_mm,
                "max_core_penetration_mm": self.arena.max_core_penetration_mm,
                "worst_contact_pair": self.arena.worst_contact_pair,
                "food_balance_error": float(self.environment.initial_food + self.environment.refilled_food
                    - self.environment.withdrawn_food - self.environment.ingested_food - remaining),
                "contestants": rows}

    def close(self):
        self.arena.close()


def scene(patch_radius=1.4, amount=40.0, energy=14.0, downwind=4.5, spread=2.5,
          swap=False, solo=False):
    """One patch in the centre, two hungry flies placed fairly.

    Placement matters more than it looks. The plume drifts with the wind, so a
    fly standing upwind of the patch measures exactly zero odour while the one
    downwind measures a lot: an earlier version of this scene put them on
    opposite sides and produced a 3-0 result that was nothing but that. Both
    flies now start downwind at the same distance, mirrored across the wind axis.

    Mirroring is still not enough on its own, because the plume is made of
    stochastic filaments and is never symmetric at any instant, so every seed is
    run twice with ``swap`` exchanging the two sides. A result that follows the
    fly across the swap means something; a result that follows the position does
    not.

    The amount is large on purpose: with an intake rate of 2 food_u per second of
    contact, a small patch empties before the second fly arrives, and the
    question of who holds it is never put.
    """
    config = load_config(Path(__file__).resolve().parents[1] / "configs/search-arena.toml")
    config["food"] = [{"id": "contested", "x": 0., "y": 0., "radius": patch_radius,
                       "amount": amount, "energy_density": 12., "taste": 1., "odor": 1.3}]
    config["initial"] = {"energy": energy, "sleep_pressure": .05,
                         "dust_by_region": dict.fromkeys(
                             ("head", "antenna_left", "antenna_right", "front_left", "front_right"), 0.)}
    left, right = (-downwind, -spread, .8), (-downwind, spread, .8)
    if solo:
        # Same world, same plume, same patch, but the rival is parked in a far
        # corner. Whatever the first fly eats here it ate without a competitor,
        # which is the only way to tell a competitive outcome from a ceiling.
        return config, [left, (-18., 18., .8)]
    return config, [right, left] if swap else [left, right]


def encounter_scene(amount=None, swap=False, energy=14.0, patch_radius=1.4, solo=False,
                    resources="scarce"):
    """A controlled near-food encounter, separate from plume-search competition.

    Both start facing the patch at equal range. It isolates interaction from
    long search latency; it does not establish success from arbitrary starts.
    """
    if resources not in {"scarce", "ample"}:
        raise ValueError("resources must be scarce or ample")
    amount = (4. if resources == "scarce" else 40.) if amount is None else amount
    config, _ = scene(amount=amount, energy=energy, patch_radius=patch_radius)
    config["duel"] = {"head_contact": True, "collision_geometry": "mesh", "resources": resources}
    starts = [(-1.65, 0., .8), (1.65, 0., .8)]
    headings = [0., float(np.pi)]
    if solo:
        starts[1] = (-18., 18., .8)
    if swap:
        starts.reverse()
        headings.reverse()
    return config, starts, headings


def run(seconds=30.0, *, seed=1, body_contact=True, output=None, swap=False, solo=False, **kwargs):
    config, start = scene(swap=swap, solo=solo, **kwargs)
    duel = Duel(config, seed=seed, body_contact=body_contact, start=start)
    try:
        for _ in range(round(seconds / .01)):
            duel.step()
        report = duel.summary()
    finally:
        duel.close()
    report["body_contact_enabled"] = body_contact
    report["seed"] = seed
    report["sides_swapped"] = swap
    report["solo_control"] = solo
    report["start_positions"] = [list(point) for point in start]
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        tag = (f"seed{seed}{'-solo' if solo else ''}{'-swapped' if swap else ''}"
               f"{'' if body_contact else '-nocontact'}")
        (output / f"duel-{tag}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return report
