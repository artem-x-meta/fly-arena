"""Explicit resource model in arbitrary food/energy units, never calibrated SI.

Only metabolism, digestion, sleep homeostasis and circadian phase use life time.
The environment owns the physical intake transaction and passes its amount to
``consume``; ``intake_rate`` is food units per *physical* second.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any


DUST_REGIONS = ("head", "antenna_left", "antenna_right", "front_left", "front_right")


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass
class SimulationClocks:
    life_time_scale: float = 1.0
    physics_time_s: float = 0.0
    neural_time_s: float = 0.0
    life_time_s: float = 0.0
    wall_time_s: float = 0.0

    def advance(self, dt_physics: float, dt_neural: float | None = None,
                *, paused: bool = False, wall_elapsed_s: float | None = None) -> float:
        dt_physics = _nonnegative("dt_physics", dt_physics)
        dt_neural = dt_physics if dt_neural is None else _nonnegative("dt_neural", dt_neural)
        if not math.isclose(dt_physics, dt_neural, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Physics and neural intervals must agree")
        if wall_elapsed_s is not None:
            self.wall_time_s = _nonnegative("wall_elapsed_s", wall_elapsed_s)
        if paused:
            return 0.0
        dt_life = dt_physics * self.life_time_scale
        self.physics_time_s += dt_physics
        self.neural_time_s += dt_neural
        self.life_time_s += dt_life
        return dt_life


@dataclass
class OrganismConfig:
    life_time_scale: float = 1.0
    energy_capacity: float = 100.0  # energy_u
    gut_capacity: float = 10.0  # food_u
    intake_rate: float = 2.0  # food_u / physical_s, applied by environment
    digestion_rate: float = .02  # food_u / life_s
    assimilation_efficiency: float = .8  # dimensionless
    tau_awake: float = 600.0  # life_s
    tau_sleep: float = 180.0  # life_s
    circadian_period: float = 86400.0  # life_s
    basal_metabolic_rate: float = .03  # energy_u / life_s
    movement_metabolic_rate: float = .02  # energy_u / life_s / movement_proxy
    action_metabolic_multipliers: dict[str, float] = field(default_factory=lambda: {
        "WALK": 2.0, "IDLE": 1.0, "FEED": 1.2, "GROOM_FRONT": 1.8,
        "GROOM_HEAD": 1.8, "SLEEP_ENTRY": 1.0, "SLEEP": .6,
        "WAKE": 1.2, "EXHAUSTED": .7,
        "ESCAPE": 3.0,
    })

    def __post_init__(self):
        for name in ("life_time_scale", "energy_capacity", "gut_capacity", "tau_awake",
                     "tau_sleep", "circadian_period"):
            if _nonnegative(name, getattr(self, name)) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("intake_rate", "digestion_rate", "basal_metabolic_rate", "movement_metabolic_rate"):
            _nonnegative(name, getattr(self, name))
        if not 0 <= self.assimilation_efficiency <= 1:
            raise ValueError("assimilation_efficiency must be in [0, 1]")
        for action, value in self.action_metabolic_multipliers.items():
            _nonnegative(f"metabolic multiplier {action}", value)


@dataclass
class FoodBolus:
    food_id: str
    amount: float  # food_u
    energy_density: float  # energy_u / food_u, retained at ingestion


@dataclass
class OrganismState:
    energy: float = 60.0
    gut_batches: list[FoodBolus] = field(default_factory=list)
    sleep_pressure: float = .2
    circadian_phase: float = 0.0  # fraction of external cycle, [0, 1)
    dust_by_region: dict[str, float] = field(default_factory=lambda: dict.fromkeys(DUST_REGIONS, 0.0))
    ingested_total: float = 0.0
    digested_total: float = 0.0
    digested_food_energy_total: float = 0.0
    assimilated_energy_total: float = 0.0
    assimilation_loss_total: float = 0.0
    metabolic_spent_total: float = 0.0
    metabolic_unmet_total: float = 0.0
    energy_overflow_total: float = 0.0

    @property
    def gut_amount(self) -> float:
        return math.fsum(b.amount for b in self.gut_batches)

    @property
    def gut_food_energy(self) -> float:
        return math.fsum(b.amount * b.energy_density for b in self.gut_batches)


@dataclass
class StepBalance:
    dt_physics: float = 0.0
    dt_life: float = 0.0
    intake: float = 0.0
    digested: float = 0.0
    food_energy: float = 0.0
    assimilated_energy: float = 0.0
    assimilation_loss: float = 0.0
    metabolic_requested: float = 0.0
    metabolic_spent: float = 0.0
    metabolic_unmet: float = 0.0
    energy_overflow: float = 0.0


class Organism:
    def __init__(self, config: OrganismConfig | dict | None = None,
                 initial: dict | OrganismState | None = None):
        self.config = OrganismConfig(**config) if isinstance(config, dict) else config or OrganismConfig()
        self.clocks = SimulationClocks(life_time_scale=self.config.life_time_scale)
        if initial is None:
            initial = {"energy": .6 * self.config.energy_capacity}
        self.state = self._state_from_dict(asdict(initial) if isinstance(initial, OrganismState) else initial)
        self._validate_state(self.state)

    @staticmethod
    def _state_from_dict(initial: dict) -> OrganismState:
        data = dict(initial)
        # Convenience for scenario initial conditions. Composition is made
        # explicit immediately and survives changes of food source.
        if "gut_amount" in data:
            amount = data.pop("gut_amount")
            density = data.pop("gut_energy_density", 0.0)
            if data.get("gut_batches"):
                raise ValueError("Specify gut_amount or gut_batches, not both")
            data["gut_batches"] = [FoodBolus("initial", amount, density)] if amount else []
        else:
            data.pop("gut_energy_density", None)
        data["gut_batches"] = [FoodBolus(**b) if isinstance(b, dict) else FoodBolus(**asdict(b))
                               for b in data.get("gut_batches", [])]
        data["dust_by_region"] = dict(dict.fromkeys(DUST_REGIONS, 0.0), **data.get("dust_by_region", {}))
        return OrganismState(**data)

    def _validate_state(self, state: OrganismState):
        if not 0 <= state.energy <= self.config.energy_capacity:
            raise ValueError("energy outside capacity")
        if not 0 <= state.sleep_pressure <= 1 or not 0 <= state.circadian_phase < 1:
            raise ValueError("sleep_pressure/phase outside range")
        for batch in state.gut_batches:
            _nonnegative("gut amount", batch.amount)
            _nonnegative("food energy density", batch.energy_density)
        if state.gut_amount > self.config.gut_capacity + 1e-10:
            raise ValueError("gut exceeds capacity")
        for value in state.dust_by_region.values():
            _nonnegative("dust", value)
        for f in fields(state):
            if f.name.endswith("_total"):
                _nonnegative(f.name, getattr(state, f.name))

    @property
    def gut_free_capacity(self) -> float:
        return max(0.0, self.config.gut_capacity - self.state.gut_amount)

    @property
    def hunger(self) -> float:
        # Both stores suppress hunger. A full gut suppresses it immediately,
        # while food energy is still waiting to be assimilated.
        energy_deficit = 1 - self.state.energy / self.config.energy_capacity
        gut_fraction = self.state.gut_amount / self.config.gut_capacity
        return max(0.0, min(1.0, energy_deficit * (1 - gut_fraction)))

    @property
    def exhausted(self) -> bool:
        return self.state.energy <= 1e-12

    def consume(self, food_id: str, amount: float, density: float) -> float:
        """Commit the exact food transfer already permitted by physical events.

        No clipping here: silently accepting less than the environment removed
        would destroy mass conservation. The caller bounds the shared amount.
        """
        amount = _nonnegative("intake amount", amount)
        density = _nonnegative("food energy density", density)
        if amount > self.gut_free_capacity + 1e-10:
            raise ValueError("Intake transaction exceeds free gut capacity")
        if amount:
            batches = self.state.gut_batches
            if batches and batches[-1].food_id == str(food_id) and batches[-1].energy_density == density:
                batches[-1].amount += amount
            else:
                batches.append(FoodBolus(str(food_id), amount, density))
            self.state.ingested_total += amount
        return amount

    def advance(self, dt_physics: float, action: str, ingestion=None,
                movement_proxy: float = 0.0, paused: bool = False) -> StepBalance:
        dt_physics = _nonnegative("dt_physics", dt_physics)
        movement_proxy = _nonnegative("movement_proxy", movement_proxy)
        if paused or dt_physics == 0:
            return StepBalance()
        # Validate the complete intake batch before any mutation.
        ingestion = list(ingestion or ())
        intake = math.fsum(_nonnegative("intake amount", b[1]) for b in ingestion)
        for _, _, density in ingestion:
            _nonnegative("food energy density", density)
        if intake > self.gut_free_capacity + 1e-10:
            raise ValueError("Intake transaction exceeds free gut capacity")
        for food_id, amount, density in ingestion:
            self.consume(food_id, amount, density)
        dt_life = self.clocks.advance(dt_physics)
        result = StepBalance(dt_physics=dt_physics, dt_life=dt_life, intake=intake)
        remaining = min(self.state.gut_amount, self.config.digestion_rate * dt_life)
        # FIFO digestion retains each swallowed source's nutritional density.
        while remaining > 0 and self.state.gut_batches:
            batch = self.state.gut_batches[0]
            amount = min(batch.amount, remaining)
            result.digested += amount
            result.food_energy += amount * batch.energy_density
            batch.amount -= amount
            remaining -= amount
            if batch.amount <= 0:
                self.state.gut_batches.pop(0)
        result.assimilated_energy = self.config.assimilation_efficiency * result.food_energy
        result.assimilation_loss = result.food_energy - result.assimilated_energy
        multiplier = self.config.action_metabolic_multipliers.get(action, 1.0)
        result.metabolic_requested = (self.config.basal_metabolic_rate * multiplier +
                                      self.config.movement_metabolic_rate * movement_proxy) * dt_life
        available = self.state.energy + result.assimilated_energy
        result.metabolic_spent = min(available, result.metabolic_requested)
        result.metabolic_unmet = result.metabolic_requested - result.metabolic_spent
        remainder = available - result.metabolic_spent
        result.energy_overflow = max(0.0, remainder - self.config.energy_capacity)
        self.state.energy = min(self.config.energy_capacity, remainder)
        if action == "SLEEP":
            self.state.sleep_pressure *= math.exp(-dt_life / self.config.tau_sleep)
        else:
            self.state.sleep_pressure = 1 - (1 - self.state.sleep_pressure) * math.exp(-dt_life / self.config.tau_awake)
        self.state.circadian_phase = (self.state.circadian_phase + dt_life / self.config.circadian_period) % 1.0
        for state_name, balance_name in (
            ("digested_total", "digested"), ("digested_food_energy_total", "food_energy"),
            ("assimilated_energy_total", "assimilated_energy"), ("assimilation_loss_total", "assimilation_loss"),
            ("metabolic_spent_total", "metabolic_spent"), ("metabolic_unmet_total", "metabolic_unmet"),
            ("energy_overflow_total", "energy_overflow"),
        ):
            setattr(self.state, state_name, getattr(self.state, state_name) + getattr(result, balance_name))
        self._validate_state(self.state)
        return result

    def get_state(self) -> dict[str, Any]:
        return {"version": 1, "config": asdict(self.config), "state": asdict(self.state), "clocks": asdict(self.clocks)}

    def set_state(self, saved: dict[str, Any]):
        if saved.get("version") != 1 or saved.get("config") != asdict(self.config):
            raise ValueError("Incompatible organism checkpoint/configuration")
        state = self._state_from_dict(saved["state"])
        self._validate_state(state)
        clocks = SimulationClocks(**saved["clocks"])
        for name, value in asdict(clocks).items():
            _nonnegative(name, value)
        if clocks.life_time_scale != self.config.life_time_scale:
            raise ValueError("Incompatible life time scale")
        if not math.isclose(clocks.physics_time_s, clocks.neural_time_s, abs_tol=1e-9):
            raise ValueError("Checkpoint physics/neural clocks disagree")
        if not math.isclose(clocks.life_time_s, clocks.physics_time_s * clocks.life_time_scale, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Checkpoint life clock disagrees")
        self.state, self.clocks = state, clocks
