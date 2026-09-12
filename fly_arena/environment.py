"""Local analytical sensory fields and conserved physical resource transfers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np

from .sensors import DUST_REGIONS, PhysicalEvents, SensorFrame


@dataclass
class FoodPatch:
    id: str
    x: float
    y: float
    radius: float = .8
    surface_z: float = .01
    amount: float = 10.0
    energy_density: float = 12.0  # energy_u / food_u
    taste: float = 1.0
    odor: float = 1.0
    odor_length_mm: float = 5.0
    odor_when_empty: bool = False

    def __post_init__(self):
        numbers = [self.x, self.y, self.radius, self.surface_z, self.amount,
                   self.energy_density, self.taste, self.odor, self.odor_length_mm]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Food parameters must be finite")
        if min(self.amount, self.energy_density, self.taste, self.odor) < 0:
            raise ValueError("Food stock, density and sensory intensities cannot be negative")
        if self.radius <= 0 or self.odor_length_mm <= 0:
            raise ValueError("Food radius and odor length must be positive")

    def overlaps(self, point, thickness):
        return (math.hypot(point[0] - self.x, point[1] - self.y) <= self.radius
                and abs(float(point[2]) - self.surface_z) <= thickness)


class Environment:
    def __init__(self, food_patches=(), config=None, events=(), *, seed=1):
        self.config = {"contact_thickness_mm": .04, "cleaning_enabled": True,
                       "mouth_contact_enabled": True, "dust_per_mm": .8,
                       "max_cleaning_per_physics_s": 1.5, "head_transfer_fraction": .7,
                       "dust_deposition_per_life_s": 0.0, "light": 1.0}
        self.config.update(config or {})
        self.food = [FoodPatch(**item) if isinstance(item, dict) else item for item in food_patches]
        if len({p.id for p in self.food}) != len(self.food):
            raise ValueError("Food IDs must be unique")
        if not 0 <= self.config["head_transfer_fraction"] <= 1:
            raise ValueError("head_transfer_fraction must lie in [0, 1]")
        self.events = sorted(copy.deepcopy(list(events)), key=lambda e: e["time_s"])
        self.event_cursor = 0
        self.recurring_events = copy.deepcopy(self.config.get("recurring_events", []))
        for event in self.recurring_events:
            if event.get("repeat_s", 0) <= 0 or event.get("time_s", 0) < 0:
                raise ValueError("Recurring events require repeat_s>0 and time_s>=0")
            event["next_time_s"] = float(event.get("time_s", 0))
            event["occurrences"] = 0
        self.active_stimuli = []
        self.initial_food = sum(p.amount for p in self.food)
        self.refilled_food = self.withdrawn_food = self.ingested_food = 0.0
        self.removed_dust = self.added_dust = 0.0
        self.last_transfers = {}
        self.event_log = []
        self.initial_dust = None
        self.plume = None
        if self.config.get("odor_model", "analytic") == "filaments":
            from .plume import OdorPlume
            self.plume = OdorPlume(self.food, self.config.get("plume", {}), seed)
        elif self.config.get("odor_model", "analytic") != "analytic":
            raise ValueError("Unknown odor model")
        from .escape import LoomingEvents
        self.looming = LoomingEvents(self.config.get("looming", []))

    def advance_field(self, dt):
        if self.plume is not None:
            self.plume.advance(self.food, dt)

    def odor_at(self, point):
        if self.plume is not None:
            return self.plume.sample(point)
        return sum(p.odor * math.exp(-math.hypot(point[0] - p.x, point[1] - p.y) / p.odor_length_mm)
                   for p in self.food if p.amount > 0 or p.odor_when_empty)

    def taste_at(self, point):
        return max((p.taste for p in self.food if p.amount > 0 and
                    p.overlaps(point, self.config["contact_thickness_mm"])), default=0.0)

    def wake_at(self, point, time_s):
        return sum(e["amplitude"] * max(0.0, 1 - np.linalg.norm(np.asarray(point) - e["position"]) / e["radius"])
                   for e in self.active_stimuli if e["start_s"] <= time_s < e["end_s"])

    def add_wake(self, time_s, position, amplitude=1.0, duration_s=.1, radius=5.0):
        if duration_s <= 0 or radius <= 0 or amplitude < 0:
            raise ValueError("Wake pulse requires positive duration/radius and nonnegative amplitude")
        self.active_stimuli.append({"position": list(position), "amplitude": amplitude,
                                    "radius": radius, "start_s": time_s, "end_s": time_s + duration_s})

    def apply_scheduled_events(self, time_s, organism):
        if self.initial_dust is None:
            self.initial_dust = sum(organism.state.dust_by_region.values())
        self.active_stimuli = [e for e in self.active_stimuli if e["end_s"] > time_s]
        emitted = []
        emitted.extend(self.looming.poll(time_s))
        due_events = []
        while self.event_cursor < len(self.events) and self.events[self.event_cursor]["time_s"] <= time_s + 1e-9:
            due_events.append(copy.deepcopy(self.events[self.event_cursor]))
            self.event_cursor += 1
        for scheduled in self.recurring_events:
            while scheduled["next_time_s"] <= time_s + 1e-9:
                event = copy.deepcopy(scheduled)
                event["time_s"] = scheduled["next_time_s"]
                due_events.append(event)
                scheduled["occurrences"] += 1
                scheduled["next_time_s"] = scheduled.get("time_s", 0) + scheduled["occurrences"] * scheduled["repeat_s"]
        for event in sorted(due_events, key=lambda e: e["time_s"]):
            kind = event["kind"]
            if kind == "wake":
                self.add_wake(time_s, event.get("position", [0, 0, 1]), event.get("amplitude", 1),
                              event.get("duration_s", .1), event.get("radius", 30))
            elif kind in ("refill", "remove_food"):
                patch = next(p for p in self.food if p.id == event["food_id"])
                if kind == "refill":
                    amount = float(event["amount"])
                    if amount < 0:
                        raise ValueError("Refill must not be negative")
                    if "capacity" in event:
                        amount = min(amount, max(0, event["capacity"] - patch.amount))
                    patch.amount += amount
                    self.refilled_food += amount
                    event["amount_added"] = amount
                else:
                    self.withdrawn_food += patch.amount
                    patch.amount = 0.0
            elif kind == "dust":
                amount = float(event["amount"])
                if amount < 0 or event["region"] not in DUST_REGIONS:
                    raise ValueError("Invalid dust event")
                organism.state.dust_by_region[event["region"]] += amount
                self.added_dust += amount
            elif kind in ("cleaning_enabled", "mouth_contact_enabled", "light"):
                self.config[kind] = event["value"]
            else:
                raise ValueError(f"Unknown external event {kind!r}")
            event["applied_time_s"] = time_s
            emitted.append(event)
        return emitted

    def sense(self, arena, organism, eyes=None):
        time_s = organism.clocks.physics_time_s
        antennas = arena.antenna_positions
        if isinstance(antennas, dict):
            antennas = list(antennas.values())
        tarsi = arena.tarsal_positions
        if not isinstance(tarsi, dict):
            tarsi = dict(zip(("LF", "LM", "LH", "RF", "RM", "RH"), tarsi))
        mouth = arena.mouth_position
        dust = {key: float(organism.state.dust_by_region.get(key, 0)) for key in DUST_REGIONS}
        mouth_contact = any(p.amount > 0 and p.overlaps(mouth, self.config["contact_thickness_mm"]) for p in self.food)
        wind_body = np.zeros(2)
        if self.plume is not None:
            rotation = arena.sim.mj_data.xmat[arena.thorax_id].reshape(3, 3)
            wind_body = rotation[:2, :2].T @ self.plume.wind
        return SensorFrame(time_s=time_s, vision=eyes,
                           odor_left=self.odor_at(antennas[0]), odor_right=self.odor_at(antennas[1]),
                           tarsal_taste={key: self.taste_at(pos) for key, pos in tarsi.items()},
                           mouth_taste=self.taste_at(mouth), dust_afferents=dust,
                           tactile_by_region=dust.copy(), upright=arena.upright,
                           ground_support=int(getattr(arena, "ground_support", 6)),
                           movement=float(getattr(arena, "movement", 0)),
                           local_wake_stimulus=self.wake_at(arena.position, time_s),
                           mouth_contact=mouth_contact,
                           wind_body_x=float(wind_body[0]), wind_body_y=float(wind_body[1]))

    def apply_physical_events(self, events: PhysicalEvents, organism, dt_physics):
        if self.initial_dust is None:
            self.initial_dust = sum(organism.state.dust_by_region.values())
        if dt_physics <= 0:
            raise ValueError("Physical interval must be positive")
        durations = events.contact_s_by_food
        if any(not math.isfinite(v) or v < 0 for v in durations.values()) or sum(durations.values()) > dt_physics + 1e-9:
            raise ValueError("Permitted mouth contact time exceeds the physical interval")
        intake_by_food = {}
        for patch in self.food:
            duration = durations.get(patch.id, 0.0) if self.config["mouth_contact_enabled"] else 0.0
            amount = min(patch.amount, organism.gut_free_capacity, organism.config.intake_rate * duration)
            if amount > 0:
                # A single transaction: this exact amount enters the gut.
                organism.consume(patch.id, amount, patch.energy_density)
                patch.amount -= amount
                self.ingested_food += amount
                intake_by_food[patch.id] = amount
        cleaned = {}
        if self.config["cleaning_enabled"]:
            dust = organism.state.dust_by_region
            for pair, sliding in events.grooming_sliding_by_pair.items():
                if sliding < 0 or not math.isfinite(sliding):
                    raise ValueError("Invalid physical sliding distance")
                contact_s = events.grooming_contact_s_by_pair.get(pair, 0.0)
                if not 0 <= contact_s <= dt_physics + 1e-9:
                    raise ValueError("Invalid grooming contact duration")
                capacity = min(self.config["dust_per_mm"] * sliding,
                               self.config["max_cleaning_per_physics_s"] * contact_s)
                a, b = pair.split("|")
                if a not in DUST_REGIONS or b not in DUST_REGIONS:
                    raise ValueError("Unknown cleaning region")
                if {a, b} == {"front_left", "front_right"}:
                    total = dust[a] + dust[b]
                    amount = min(total, capacity)
                    if total > 0:
                        dust[a] -= amount * dust[a] / total
                        dust[b] -= amount * dust[b] / total
                    self.removed_dust += amount
                else:
                    leg, head = (a, b) if a.startswith("front_") else (b, a)
                    if not leg.startswith("front_") or head.startswith("front_"):
                        continue
                    amount = min(dust[head], capacity)
                    transferred = amount * self.config["head_transfer_fraction"]
                    dust[head] -= amount
                    dust[leg] += transferred
                    self.removed_dust += amount - transferred
                cleaned[pair] = amount
        deposition = self.config["dust_deposition_per_life_s"] * organism.config.life_time_scale * dt_physics
        if deposition:
            if deposition < 0:
                raise ValueError("Dust deposition rate cannot be negative")
            organism.state.dust_by_region["head"] += deposition
            self.added_dust += deposition
        self.last_transfers = {"intake_by_food": intake_by_food, "cleaned_by_pair": cleaned}
        self.check_balances(organism)
        return self.last_transfers

    def check_balances(self, organism):
        food_residual = self.initial_food + self.refilled_food - self.withdrawn_food - self.ingested_food - sum(p.amount for p in self.food)
        dust_residual = (self.initial_dust or 0) + self.added_dust - self.removed_dust - sum(organism.state.dust_by_region.values())
        if abs(food_residual) > 1e-7 or abs(dust_residual) > 1e-7:
            raise ArithmeticError(f"Resource balance violated: food={food_residual}, dust={dust_residual}")
        return {"food_residual": food_residual, "dust_residual": dust_residual}

    def get_state(self):
        return copy.deepcopy({"food": [asdict(p) for p in self.food], "config": self.config,
                              "plume": self.plume.get_state() if self.plume is not None else None,
                              "looming": self.looming.get_state(),
                              "events": self.events, "event_cursor": self.event_cursor,
                              "recurring_events": self.recurring_events,
                              "active_stimuli": self.active_stimuli,
                              **{name: getattr(self, name) for name in (
                                  "initial_food", "refilled_food", "withdrawn_food", "ingested_food",
                                  "removed_dust", "added_dust", "initial_dust", "last_transfers")}})

    def set_state(self, state):
        state = copy.deepcopy(state)
        self.food = [FoodPatch(**p) for p in state.pop("food")]
        plume_state = state.pop("plume", None)
        looming_state = state.pop("looming", {"objects": [], "last_cycles": []})
        from .escape import LoomingEvents
        self.looming = LoomingEvents(looming_state["objects"])
        self.looming.set_state(looming_state)
        if plume_state is not None:
            from .plume import OdorPlume
            if self.plume is None:
                self.plume = OdorPlume(self.food, plume_state["config"])
            self.plume.set_state(plume_state)
        else:
            self.plume = None
        for name, value in state.items():
            setattr(self, name, value)
