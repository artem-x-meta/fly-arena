"""Bounded engineered proximity and actual body-contact observations.

Proximity uses relative thorax geometry inside a short radius. It approximates
local perception: it is neither rendered vision nor reconstructed sensory
neurons. Touch is supplied by the arena's actual MuJoCo contact constraints.
Only this scalar frame crosses into the policy; opponent state and opponent
world coordinates do not. Own position is explicit engineered proprioception
for returning to a location where the animal previously tasted food.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SocialFrame:
    detected: bool = False
    bearing: float = 0.0
    distance_mm: float | None = None
    body_contact: bool = False
    contact_impulse: float = 0.0
    own_heading: float = 0.0
    own_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


def observe_social_frame(*, own_position, own_heading, rival_position,
                         range_mm=3.0, enabled=True, body_contact=False,
                         contact_impulse=0.0, contact_position=None):
    """Observe one opponent locally without reading its behavior or physiology.

    ``contact_impulse`` is the integrated norm of contact force in MuJoCo's
    model units times seconds over the preceding exchange. It is not a newly
    applied impulse. A touch can report direction beyond the proximity radius,
    but only proximity supplies a distance estimate. Disabling the sensor
    suppresses both opponent channels and retains own orientation and position.
    """
    radius = float(range_mm)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("Social sensor range must be finite and non-negative")
    heading = float(own_heading)
    own = np.asarray(own_position, dtype=float)
    own_location = tuple(float(value) for value in own)
    if not enabled:
        return SocialFrame(own_heading=heading, own_position_mm=own_location)
    delta = np.asarray(rival_position, dtype=float) - own
    distance = float(np.linalg.norm(delta))
    proximity = radius > 0.0 and distance <= radius
    touching = bool(body_contact)
    if not proximity and not touching:
        return SocialFrame(own_heading=heading, own_position_mm=own_location)
    # A physical touch supplies its own location even when proximity is off.
    if touching and contact_position is not None:
        delta = np.asarray(contact_position, dtype=float) - own
    angle = float(np.arctan2(delta[1], delta[0]) - heading)
    bearing = float(np.arctan2(np.sin(angle), np.cos(angle)))
    return SocialFrame(
        detected=True, bearing=bearing,
        distance_mm=distance if proximity else None,
        body_contact=touching,
        contact_impulse=max(0.0, float(contact_impulse)) if touching else 0.0,
        own_heading=heading,
        own_position_mm=own_location,
    )
