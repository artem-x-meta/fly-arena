"""Transparent need arbitration. Durations belong to actions, not a life script.

This module receives local senses, organism state and neural proposals only.
It has no environment reference, food coordinates or target-location API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import math


DEFAULTS = {
    "min_action_duration_s": .35,
    "upright_min": .8,
    # A tripod stance is the physical minimum while walking; the arena reports
    # the same figure in its own stability property. Sleep keeps the stricter
    # requirement, because a settled resting pose stands on more legs.
    "support_min": 3,
    "sleep_support_min": 4,
    "sleep_enter_threshold": .72,
    "sleep_exit_threshold": .28,
    "sleep_entry_quiet_s": .6,
    "sleep_entry_timeout_s": 4.0,
    "sleep_min_duration_s": .5,
    "sleep_movement_max": .5,
    "awake_stimulus_threshold": .25,
    "sleep_stimulus_threshold": .8,
    "wake_delay_s": .25,
    "critical_energy_fraction": .08,
    "feed_hunger_enter": .25,
    "feed_hunger_exit": .1,
    "taste_threshold": .15,
    "feed_contact_timeout_s": 1.8,
    # Physical seconds from FEED entry. The mouth extends/settles for .6 s;
    # overlap while leaving a different pose cannot count as established feeding.
    "feed_contact_grace_s": .65,
    "feed_lost_contact_timeout_s": .25,
    "feed_max_duration_s": 8.0,
    "feed_retry_delay_s": .5,
    "dust_enter_threshold": .15,
    "dust_exit_threshold": .04,
    "neural_request_threshold": .1,
    "neural_walk_threshold": .01,
    "feed_weight": 1.6,
    "walk_hunger_weight": .35,
    "feed_preempts_walk": False,
    "groom_front_weight": 1.1,
    "groom_head_weight": 1.0,
    "sleep_weight": 1.3,
    "circadian_rest_weight": .12,
    "switch_margin": .1,
    "allow_experimental_ports": False,
    "escape_enabled": False,
    "escape_threshold": 1.0,
    "escape_sleep_threshold": 2.0,
    "escape_duration_s": 1.4,
    "escape_refractory_s": 2.0,
    "escape_upright_min": .85,
    "escape_support_grace_s": .04,
}


@dataclass
class BehaviorDecision:
    action: str
    target_body_region: str | None = None
    reason: str = ""
    source: str = "engineered_policy"
    gates: dict[str, object] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    escape_turn: float = 0.0


def _supported(readout, port: str, allow_experimental=False) -> bool:
    value = readout.supported_ports.get(port, False)
    if isinstance(value, dict):
        value = value.get("status", False)
    # A candidate annotation is deliberately insufficient.
    return value is True or value in ("confirmed", "supported", "validated") or (allow_experimental and value == "experimental")


class BehaviorController:
    def __init__(self, config: dict | None = None, seed: int = 1):
        self.config = dict(DEFAULTS)
        unknown = set(config or {}) - self.config.keys()
        if unknown:
            raise ValueError(f"Unknown behavior settings: {sorted(unknown)}")
        self.config.update(config or {})
        for key, value in self.config.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0 <= self.config["sleep_exit_threshold"] < self.config["sleep_enter_threshold"] <= 1:
            raise ValueError("Sleep thresholds must satisfy 0 <= exit < enter <= 1")
        if not 0 <= self.config["feed_hunger_exit"] < self.config["feed_hunger_enter"] <= 1:
            raise ValueError("Feeding thresholds must satisfy 0 <= exit < enter <= 1")
        if self.config["awake_stimulus_threshold"] >= self.config["sleep_stimulus_threshold"]:
            raise ValueError("Sleeping stimulus threshold must exceed awake threshold")
        if self.config["wake_delay_s"] <= 0 or self.config["sleep_entry_quiet_s"] <= 0:
            raise ValueError("Sleep entry and wake transitions require positive duration")
        if self.config["feed_contact_grace_s"] >= self.config["feed_contact_timeout_s"]:
            raise ValueError("Feeding contact grace must end before the attempt timeout")
        if self.config["sleep_support_min"] < self.config["support_min"]:
            raise ValueError("Sleep support requirement cannot be weaker than the stability guard")
        self.seed = int(seed)  # No RNG is needed for deterministic need arbitration.
        self.action = "IDLE"
        self.target_body_region = None
        self.action_elapsed_s = 0.0
        self.quiet_elapsed_s = 0.0
        self.feed_retry_remaining_s = 0.0
        self.feed_contact_seen = False
        self.feed_lost_contact_s = 0.0
        self.transition_count = 0
        self.last_transition = None  # Streaming owner can emit this on count changes.
        self.last_decision = BehaviorDecision("IDLE", reason="initial_state")
        self.escape_cooldown_s = 0.0
        self.escape_turn = 0.0
        self.pending_escape = False
        self.pending_escape_source = "engineered_policy"
        self.escape_unsupported_s = 0.0

    def _choose(self, action, reason, source, gates, scores, time_s, target=None):
        changed = action != self.action or target != self.target_body_region
        decision = BehaviorDecision(action, target, reason, source, dict(gates), dict(scores),
                                    self.escape_turn if action == "ESCAPE" else 0.0)
        if changed:
            previous = self.action
            if previous == "ESCAPE" and action != "ESCAPE":
                self.escape_cooldown_s = self.config["escape_refractory_s"]
            self.action = action
            self.target_body_region = target
            self.action_elapsed_s = 0.0
            if action == "SLEEP_ENTRY":
                self.quiet_elapsed_s = 0.0
            if action == "FEED":
                self.feed_contact_seen = False
                self.feed_lost_contact_s = 0.0
            self.transition_count += 1
            self.last_transition = {"time_s": time_s, "previous_action": previous,
                                    "transition_count": self.transition_count, **asdict(decision)}
        self.last_decision = decision
        return decision

    def decide(self, sensor_frame, organism, neural_readout, dt=.01, mode="ethology-hybrid"):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Behavior interval must be positive and finite")
        if mode not in ("ethology-hybrid", "ethology-neural"):
            raise ValueError("Behavior mode must be ethology-hybrid or ethology-neural")
        c, s, o, n = self.config, sensor_frame, organism, neural_readout
        self.action_elapsed_s += dt
        self.feed_retry_remaining_s = max(0.0, self.feed_retry_remaining_s - dt)
        self.escape_cooldown_s = max(0.0, self.escape_cooldown_s - dt)
        hybrid = mode == "ethology-hybrid"
        source = "engineered_policy" if hybrid else "neural_readout"
        stable = s.upright >= c["upright_min"] and s.ground_support >= c["support_min"]
        if self.action == "ESCAPE":
            self.escape_unsupported_s = 0. if s.ground_support >= c["support_min"] else self.escape_unsupported_s + dt
            stable = s.upright >= c["escape_upright_min"] and self.escape_unsupported_s <= c["escape_support_grace_s"]
        else:
            self.escape_unsupported_s = 0.0
        quiet = (stable and s.ground_support >= c["sleep_support_min"]
                 and s.local_wake_stimulus < c["awake_stimulus_threshold"]
                 and s.movement <= c["sleep_movement_max"])
        hungry = o.hunger
        pressure = o.state.sleep_pressure
        critical = o.state.energy / o.config.energy_capacity <= c["critical_energy_fraction"] and o.state.gut_amount <= 1e-10
        taste = max([s.mouth_taste, *s.tarsal_taste.values()])
        # Contact chemosensation alone admits a feeding attempt, and only the
        # front tarsi and the mouth count: they are the contacts that lie under
        # the extended proboscis. Odor is a distance cue that drives the local
        # search controller; an attempt started on odor, or on a hind tarsus
        # trailing behind the body, probes the floor away from the patch.
        front_taste = max([s.mouth_taste, *(value for key, value in s.tarsal_taste.items()
                                            if key.upper() in ("LF", "RF"))])
        local_food = front_taste >= c["taste_threshold"]
        dust = s.dust_afferents
        front_dust = sum(dust.get(region, 0.0) for region in ("front_left", "front_right"))
        head_regions = ("head", "antenna_left", "antenna_right")
        head_target = max(head_regions, key=lambda region: dust.get(region, 0.0))
        head_dust = sum(dust.get(region, 0.0) for region in head_regions)
        experimental = c["allow_experimental_ports"]
        feed_request = hybrid or (_supported(n, "feed", experimental) and n.feed_drive >= c["neural_request_threshold"])
        front_request = hybrid or (_supported(n, "groom_front", experimental) and n.groom_drives.get("front", n.groom_drives.get("front_leg_rub", 0.0)) >= c["neural_request_threshold"])
        head_request = hybrid or (_supported(n, "groom_head", experimental) and n.groom_drives.get("head", n.groom_drives.get("head_sweep", 0.0)) >= c["neural_request_threshold"])
        # Sleep homeostasis remains an explicit organism model in both modes;
        # it is never labelled as a reconstructed neural sleep circuit.
        sleep_request = True
        gates = {"stable": stable, "quiet": quiet, "local_food": local_food,
                 "critical_energy": critical, "hunger": hungry, "sleep_pressure": pressure,
                 "feed_request": feed_request, "groom_front_request": front_request,
                 "groom_head_request": head_request, "sleep_request": sleep_request,
                 "feed_retry_remaining_s": self.feed_retry_remaining_s}
        # A failed reach does not cancel the need: while the retry window is open
        # the same feeding drive expresses as approach, so a hungry fly steps
        # onto the patch instead of probing the floor from its edge.
        approaching = self.feed_retry_remaining_s > 0 and local_food and hungry >= c["feed_hunger_enter"]
        gates["feed_approach"] = approaching
        scores = {"IDLE": .05,
                  "WALK": .18 + c["walk_hunger_weight"] * hungry + (c["feed_weight"] * hungry if approaching else 0.)
                  if hybrid else max(n.walk_left, n.walk_right),
                  "FEED": c["feed_weight"] * hungry + min(taste, 1.0) * .2,
                  "GROOM_FRONT": c["groom_front_weight"] * front_dust,
                  "GROOM_HEAD": c["groom_head_weight"] * head_dust,
                  "SLEEP_ENTRY": c["sleep_weight"] * pressure +
                  c["circadian_rest_weight"] * pressure * (.5 + .5 * math.cos(2 * math.pi * o.state.circadian_phase))}

        def choose(action, reason, selected_source=source, target=None):
            if action in ("SLEEP_ENTRY", "SLEEP"):
                selected_source = "engineered_policy"
            return self._choose(action, reason, selected_source, gates, scores, s.time_s, target)

        # Physical safety and lost neural requests override minimum duration.
        if not stable:
            self.pending_escape = False
            return choose("WAKE" if self.action in ("SLEEP", "WAKE") else "IDLE",
                          "support_or_upright_guard", "physical_guard")
        if c["escape_enabled"]:
            escape_level = max(s.looming_left, s.looming_right) if hybrid else (
                n.escape_drive if _supported(n, "escape", experimental) else 0.0)
            threshold = c["escape_sleep_threshold"] if self.action == "SLEEP" else c["escape_threshold"]
            gates.update(escape_level=escape_level, escape_threshold=threshold,
                         escape_cooldown_s=self.escape_cooldown_s)
            if self.action == "ESCAPE":
                if o.exhausted:
                    return choose("EXHAUSTED", "escape_energy_depleted", "physical_guard")
                if self.action_elapsed_s >= c["escape_duration_s"]:
                    return choose("IDLE", "escape_burst_complete", "physical_guard")
                return choose("ESCAPE", "escape_motor_burst", self.pending_escape_source)
            if self.pending_escape and self.action == "WAKE":
                if self.action_elapsed_s >= c["wake_delay_s"]:
                    self.pending_escape = False
                    if not o.exhausted:
                        return choose("ESCAPE", "visual_wake_complete", self.pending_escape_source)
                return choose("WAKE", "visual_wake_delay", "physical_guard")
            if escape_level >= threshold and self.escape_cooldown_s == 0 and not o.exhausted:
                self.escape_turn = (-1.0 if s.looming_left >= s.looming_right else 1.0) if hybrid else n.escape_turn
                self.pending_escape_source = source
                self.feed_retry_remaining_s = c["feed_retry_delay_s"]
                if self.action == "SLEEP":
                    self.pending_escape = True
                    return choose("WAKE", "visual_threat_wakes", "physical_guard")
                return choose("ESCAPE", "visual_looming_request", source)
        if self.action == "SLEEP":
            if s.local_wake_stimulus >= c["sleep_stimulus_threshold"]:
                return choose("WAKE", "strong_local_stimulus", "physical_guard")
            if critical:
                return choose("WAKE", "critical_energy_wakes", "physical_guard")
            if self.action_elapsed_s >= c["sleep_min_duration_s"] and pressure <= c["sleep_exit_threshold"]:
                return choose("WAKE", "sleep_pressure_relieved", "physical_guard")
            return choose("SLEEP", "sleep_hysteresis")
        if self.action == "WAKE":
            if self.action_elapsed_s < c["wake_delay_s"]:
                return choose("WAKE", "wake_transition_delay", "physical_guard")
            return choose("IDLE", "wake_transition_complete", "physical_guard")
        if self.action == "SLEEP_ENTRY":
            if not sleep_request or critical or s.local_wake_stimulus >= c["awake_stimulus_threshold"]:
                self.quiet_elapsed_s = 0.0
                return choose("IDLE", "sleep_entry_interrupted", "physical_guard")
            if pressure < c["sleep_exit_threshold"]:
                return choose("IDLE", "sleep_need_receded", "physical_guard")
            self.quiet_elapsed_s = self.quiet_elapsed_s + dt if quiet else 0.0
            if self.quiet_elapsed_s >= c["sleep_entry_quiet_s"]:
                return choose("SLEEP", "stable_quiet_sleep_entry")
            if self.action_elapsed_s >= c["sleep_entry_timeout_s"]:
                return choose("IDLE", "sleep_pose_did_not_settle", "physical_guard")
            return choose("SLEEP_ENTRY", "settling_sleep_pose")
        if self.action == "FEED":
            contact_ready = self.action_elapsed_s >= c["feed_contact_grace_s"]
            gates["feeding_contact_ready"] = contact_ready
            if contact_ready and s.mouth_contact:
                self.feed_contact_seen = True
                self.feed_lost_contact_s = 0.0
            elif contact_ready and self.feed_contact_seen:
                self.feed_lost_contact_s += dt
            stop = None
            if not feed_request:
                stop = "neural_feed_request_lost"
            elif hungry <= c["feed_hunger_exit"] or o.gut_free_capacity <= 1e-10:
                stop = "satiety_or_full_gut"
            elif contact_ready and s.mouth_contact and s.mouth_taste < c["taste_threshold"]:
                stop = "mouth_taste_unacceptable"
            elif not self.feed_contact_seen and self.action_elapsed_s >= c["feed_contact_timeout_s"]:
                stop = "mouth_contact_timeout"
            elif self.feed_lost_contact_s >= c["feed_lost_contact_timeout_s"]:
                stop = "mouth_contact_lost"
            elif self.action_elapsed_s >= c["feed_max_duration_s"]:
                stop = "feeding_attempt_duration_limit"
            elif s.local_wake_stimulus >= c["awake_stimulus_threshold"]:
                stop = "feeding_disturbed"
            if stop:
                self.feed_retry_remaining_s = c["feed_retry_delay_s"]
                return choose("IDLE", stop, "physical_guard")
            return choose("FEED", "feeding_until_feedback")
        # Depletion suppresses discretionary behaviour, but not reaching food the
        # animal can already taste: an attempt that failed leaves the approach as
        # the only route out, so the retry window must not lock the fly in place.
        if o.exhausted and not (feed_request and local_food):
            return choose("EXHAUSTED", "energy_depleted_without_accessible_food", "physical_guard")

        eligible = {"IDLE"}
        if hybrid or max(n.walk_left, n.walk_right) >= c["neural_walk_threshold"]:
            eligible.add("WALK")
        if feed_request and local_food and hungry >= c["feed_hunger_enter"] and o.gut_free_capacity > 1e-10 and self.feed_retry_remaining_s == 0:
            eligible.add("FEED")
        front_threshold = c["dust_exit_threshold"] if self.action == "GROOM_FRONT" else c["dust_enter_threshold"]
        head_threshold = c["dust_exit_threshold"] if self.action == "GROOM_HEAD" else c["dust_enter_threshold"]
        if front_request and front_dust > front_threshold:
            eligible.add("GROOM_FRONT")
        if head_request and head_dust > head_threshold:
            eligible.add("GROOM_HEAD")
        if sleep_request and pressure >= c["sleep_enter_threshold"] and not critical and s.local_wake_stimulus < c["awake_stimulus_threshold"]:
            eligible.add("SLEEP_ENTRY")
        # Critical hunger wins against competing needs, but never invents a
        # feeding output in neural mode or grants a nonlocal food destination.
        if critical or o.exhausted:
            eligible -= {"SLEEP_ENTRY", "GROOM_FRONT", "GROOM_HEAD"}
            if "FEED" in eligible:
                return choose("FEED", "critical_hunger_local_food")
        # A canonical order makes ties deterministic and checkpoint independent.
        order = ("IDLE", "WALK", "FEED", "GROOM_FRONT", "GROOM_HEAD", "SLEEP_ENTRY")
        selected = max((a for a in order if a in eligible), key=scores.__getitem__)
        reason = "highest_current_need"
        fresh_food_stop = c["feed_preempts_walk"] and self.action == "WALK" and selected == "FEED"
        if fresh_food_stop:
            reason = "front_contact_stops_search"
        if self.action in eligible and self.action != selected and not fresh_food_stop:
            if self.action_elapsed_s < c["min_action_duration_s"]:
                selected, reason = self.action, "minimum_action_duration"
            elif scores[selected] < scores[self.action] + c["switch_margin"]:
                selected, reason = self.action, "score_hysteresis"
        target = head_target if selected == "GROOM_HEAD" else "front_legs" if selected == "GROOM_FRONT" else None
        # Keep a head target through its minimum dwell unless it is clean.
        if selected == self.action == "GROOM_HEAD" and self.action_elapsed_s < c["min_action_duration_s"] and dust.get(self.target_body_region, 0) > c["dust_exit_threshold"]:
            target = self.target_body_region
        return choose(selected, reason, target=target)

    def get_state(self):
        return copy.deepcopy({"version": 1, "config": self.config, "seed": self.seed,
                              "action": self.action, "target_body_region": self.target_body_region,
                              "action_elapsed_s": self.action_elapsed_s, "quiet_elapsed_s": self.quiet_elapsed_s,
                              "feed_retry_remaining_s": self.feed_retry_remaining_s,
                              "feed_contact_seen": self.feed_contact_seen,
                              "feed_lost_contact_s": self.feed_lost_contact_s,
                              "escape_cooldown_s": self.escape_cooldown_s,
                              "escape_turn": self.escape_turn, "pending_escape": self.pending_escape,
                              "pending_escape_source": self.pending_escape_source,
                              "escape_unsupported_s": self.escape_unsupported_s,
                              "transition_count": self.transition_count, "last_transition": self.last_transition,
                              "last_decision": asdict(self.last_decision)})
        

    def set_state(self, saved):
        saved = copy.deepcopy(saved)
        if saved.pop("version", None) != 1 or saved.pop("config", None) != self.config:
            raise ValueError("Incompatible behavior checkpoint/configuration")
        if saved["action"] not in ("IDLE", "WALK", "FEED", "GROOM_FRONT", "GROOM_HEAD", "SLEEP_ENTRY", "SLEEP", "WAKE", "EXHAUSTED", "ESCAPE"):
            raise ValueError("Invalid checkpoint action")
        for key in ("action_elapsed_s", "quiet_elapsed_s", "feed_retry_remaining_s", "feed_lost_contact_s", "escape_cooldown_s"):
            if not math.isfinite(saved[key]) or saved[key] < 0:
                raise ValueError(f"Invalid checkpoint {key}")
        saved["last_decision"] = BehaviorDecision(**saved["last_decision"])
        for key, value in saved.items():
            setattr(self, key, value)
