"""Simultaneous, conserved food transfers for a shared arena.

Only permitted physical mouth-contact durations create demand. Each patch is
resolved for all contestants together, so the last portion is not awarded by
Python loop order. Patches retain the environment's explicit ordering.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

from fly_arena.environment import Environment
from fly_arena.organism import Organism
from fly_arena.sensors import DUST_REGIONS, PhysicalEvents


def _validate_batch(events, organisms, dt):
    """Reject malformed physical intervals before any resource is mutated."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Physical interval must be finite and positive")
    if len(events) != len(organisms):
        raise ValueError("Each organism requires exactly one physical event")
    if len({id(organism) for organism in organisms}) != len(organisms):
        raise ValueError("Contestants must be distinct organisms")
    for event in events:
        permitted = event.contact_s_by_food
        if (any(not math.isfinite(v) or v < 0 for v in permitted.values())
                or math.fsum(permitted.values()) > dt + 1e-9):
            raise ValueError("Permitted mouth contact time exceeds the physical interval")
        for durations in (event.mouth_contact_s_by_food,
                          event.grooming_contact_s_by_pair):
            if any(not math.isfinite(v) or not 0 <= v <= dt + 1e-9
                   for v in durations.values()):
                raise ValueError("Invalid physical contact duration")
        for pair, sliding in event.grooming_sliding_by_pair.items():
            if not math.isfinite(sliding) or sliding < 0:
                raise ValueError("Invalid physical sliding distance")
            regions = pair.split("|")
            if len(regions) != 2 or any(region not in DUST_REGIONS for region in regions):
                raise ValueError("Unknown cleaning region")


def apply_shared_events(environment: Environment, events: list[PhysicalEvents],
                        organisms: list[Organism], dt: float) -> list[dict]:
    """Return one ``intake_by_food``/``cleaned_by_pair`` result per contestant.

    Demand is bounded by intake rate times permitted contact and free gut
    capacity, then proportionally reduced if a patch cannot meet the whole
    batch. Physical contact without permission is deliberately insufficient.

    ``initial_dust`` is a pooled ledger for this environment. Callers that run
    scheduled events before the first batch must initialize it with the sum
    across contestants before those events. Otherwise it is initialized here.
    The existing single-organism cleaning implementation is reused on shallow
    ledger views, and its dust changes are merged into the shared ledger.

    ``environment.last_transfers`` holds aggregate transfers; the returned list
    keeps each contestant's transfers separate and follows input order.
    """
    _validate_batch(events, organisms, dt)
    deposition = environment.config["dust_deposition_per_life_s"]
    if not math.isfinite(deposition) or deposition < 0:
        raise ValueError("Dust deposition rate must be finite and nonnegative")
    if not organisms:
        return []
    if environment.initial_dust is None:
        environment.initial_dust = math.fsum(
            value for organism in organisms for value in organism.state.dust_by_region.values())

    results = [{"intake_by_food": {}, "cleaned_by_pair": {}} for _ in organisms]
    aggregate_intake = {}
    for patch in environment.food:
        enabled = environment.config["mouth_contact_enabled"]
        demand = [min(organism.gut_free_capacity,
                      organism.config.intake_rate * event.contact_s_by_food.get(patch.id, 0.0))
                  if enabled else 0.0 for event, organism in zip(events, organisms)]
        total = math.fsum(demand)
        if total <= 0 or patch.amount <= 0:
            continue
        scale = min(1.0, patch.amount / total)
        allocation = [amount * scale for amount in demand]
        used = math.fsum(allocation)
        # Round symmetrically, never assigning a rounding remainder to the
        # first contestant. Any unassigned amount is at most roundoff.
        while used > patch.amount:
            scale = math.nextafter(scale, 0.0)
            allocation = [amount * scale for amount in demand]
            used = math.fsum(allocation)
        for organism, result, amount in zip(organisms, results, allocation):
            if amount > 0:
                organism.consume(patch.id, amount, patch.energy_density)
                result["intake_by_food"][patch.id] = amount
        patch.amount -= used
        environment.ingested_food += used
        if used > 0:
            aggregate_intake[patch.id] = used

    removed, added = [], []
    cleaned_totals = {}
    for organism, event, result in zip(organisms, events, results):
        ledger = copy.copy(environment)
        ledger.initial_dust = math.fsum(organism.state.dust_by_region.values())
        ledger.added_dust = ledger.removed_dust = 0.0
        cleaning = ledger.apply_physical_events(
            replace(event, contact_s_by_food={}), organism, dt)
        result["cleaned_by_pair"] = cleaning["cleaned_by_pair"]
        removed.append(ledger.removed_dust)
        added.append(ledger.added_dust)
        for pair, amount in result["cleaned_by_pair"].items():
            cleaned_totals.setdefault(pair, []).append(amount)
    environment.removed_dust += math.fsum(removed)
    environment.added_dust += math.fsum(added)
    environment.last_transfers = {
        "intake_by_food": aggregate_intake,
        "cleaned_by_pair": {pair: math.fsum(amounts) for pair, amounts in cleaned_totals.items()},
    }
    remaining_dust = math.fsum(
        value for organism in organisms for value in organism.state.dust_by_region.values())
    dust_residual = (environment.initial_dust + environment.added_dust
                     - environment.removed_dust - remaining_dust)
    if abs(dust_residual) > 1e-7:
        raise ArithmeticError(f"Shared dust resource balance violated: {dust_residual}")
    return results
