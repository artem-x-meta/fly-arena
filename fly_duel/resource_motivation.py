"""Engineered motivation from a fly's own access to a recently tasted resource.

This is not a physiological scarcity detector. A fly cannot inspect a patch's
remaining budget or another fly's stomach. Equal local sensory/intake histories
therefore produce equal decisions even when the world contains different food
amounts. Short failures to place the proboscis are given time to resolve before
they can become a social engagement. All times below are physical seconds and
are explicit engineering choices, not parameters fitted to an animal study.
"""
from __future__ import annotations

import math


class ResourceMotivation:
    def __init__(self, *, feeding_opportunity_s=1.5, resource_memory_s=7.,
                 frustration_time_s=.75, intake_tau_s=.4,
                 interruption_tolerance_s=.6):
        for name, value in (("feeding_opportunity_s", feeding_opportunity_s),
                            ("resource_memory_s", resource_memory_s),
                            ("frustration_time_s", frustration_time_s),
                            ("intake_tau_s", intake_tau_s),
                            ("interruption_tolerance_s", interruption_tolerance_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, float(value))
        self.recent_resource_s = 0.
        self.opportunity_remaining_s = 0.
        self.intake_hold_s = 0.
        self.own_intake_ema = 0.  # food_u / physical_s
        self.frustration = 0.  # bounded engineering accumulator, not emotion
        self.can_engage = False
        self.reason = "no_resource"
        self._last_ingested_total = None

    def step(self, sense, local, organism, *, dt=.01, base_action="WALK") -> bool:
        """Observe only local taste/rival presence and the fly's own organism.

        The first observation establishes the cumulative intake baseline;
        historical consumption from before this controller existed is not a
        fresh meal. WALK and FEED are local access attempts. Grooming, sleeping
        and other ordinary priorities do not create denied-access frustration.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Resource interval must be positive and finite")
        ingested = float(organism.state.ingested_total)
        if not math.isfinite(ingested) or ingested < 0:
            raise ValueError("Own cumulative intake must be finite and nonnegative")
        previous = self._last_ingested_total
        if previous is not None and ingested < previous - 1e-10:
            raise ValueError("Own cumulative intake decreased; restore controller state too")
        delta = 0. if previous is None else max(0., ingested - previous)
        self._last_ingested_total = ingested
        alpha = -math.expm1(-dt / self.intake_tau_s)
        self.own_intake_ema += alpha * (delta / dt - self.own_intake_ema)

        had_resource = self.recent_resource_s > 0.
        taste = max([float(local.mouth_taste), *local.tarsal_taste.values()])
        if taste >= .15 or delta > 1e-10:
            self.recent_resource_s = self.resource_memory_s
            if not had_resource:
                self.opportunity_remaining_s = self.feeding_opportunity_s
        else:
            self.recent_resource_s = max(0., self.recent_resource_s - dt)

        # Actual swallowed food, not a FEED command, confirms successful access.
        # A short gap covers the intermittent physical mouth contact of a meal.
        self.intake_hold_s = (self.interruption_tolerance_s if delta > 1e-10
                              else max(0., self.intake_hold_s - dt))
        nominal_rate = float(organism.config.intake_rate)
        adequate_rate = max(.01, .15 * nominal_rate)
        receiving_food = (delta > 1e-10 or self.intake_hold_s > 0.
                          or self.own_intake_ema >= adequate_rate)
        needs_food = (organism.hunger >= .25 and organism.gut_free_capacity > .05
                      and not organism.exhausted)
        access_attempt = base_action in {"FEED", "WALK"}

        # Count the initial opportunity only during actual food-seeking actions.
        # A period spent asleep or grooming must not consume it.
        grace_before = self.opportunity_remaining_s
        if access_attempt and self.recent_resource_s > 0.:
            self.opportunity_remaining_s = max(0., grace_before - dt)
        unprotected_dt = max(0., dt - grace_before)

        if self.recent_resource_s <= 0.:
            self.reason = "no_recent_resource"
        elif not needs_food:
            self.reason = "own_need_satisfied"
        elif not sense.detected:
            self.reason = "no_local_rival"
        elif not access_attempt:
            self.reason = "other_own_priority"
        elif receiving_food:
            self.reason = "own_intake_successful"
        elif unprotected_dt <= 0.:
            self.reason = "initial_feeding_opportunity"
        else:
            # Hunger gates this rule but does not let a hungrier fly skip the
            # chance to feed peacefully. No identity/winner preference exists.
            self.frustration = min(1., self.frustration +
                                   unprotected_dt / self.frustration_time_s)
            self.can_engage = self.frustration >= 1. - 1e-12
            self.reason = ("own_access_failed_near_rival" if self.can_engage
                           else "waiting_for_own_access")
            return self.can_engage

        self.can_engage = False
        # Positive intake cancels a pending attack immediately. Otherwise the
        # memory of failed access decays completely in one physical second.
        self.frustration = (0. if receiving_food else max(0., self.frustration - dt))
        return False

    def summary(self):
        return {"can_engage": self.can_engage,
                "reason": self.reason,
                "frustration": self.frustration,
                "own_intake_ema_food_u_s": self.own_intake_ema,
                "recent_resource_s": self.recent_resource_s,
                "feeding_opportunity_remaining_s": self.opportunity_remaining_s,
                "intake_interruption_tolerance_s": self.intake_hold_s}
