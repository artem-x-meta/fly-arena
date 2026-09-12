"""Opponent-only collisions on the existing articulated anatomical meshes.

MuJoCo uses the convex hull of each mesh. Assign masks BEFORE compiling the
model, so body-level broadphase masks include the new geometry. Existing mass,
inertia, self contacts and explicit foot/floor contacts remain unchanged.
"""
from __future__ import annotations

import mujoco as mj


COLLISION_BITS = (1 << 8, 1 << 9)


def configure_mesh_collisions(fly, index, *, enabled=True):
    own_bit, other_bit = COLLISION_BITS[index], COLLISION_BITS[1 - index]
    names = []
    for geoms in fly.bodyseg_to_mjcfgeom.values():
        for geom in geoms:
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            names.append(f"{fly.name}/{geom.name}")
            geom.contype = own_bit if enabled else 0
            geom.conaffinity = other_bit if enabled else 0
            geom.condim = 3
            geom.friction = [.4, .001, .0001]
            geom.solref = [.0005, 1.]
            geom.solimp = [.95, .99, .001, .5, 2.]
            geom.margin = .002
    if not names:
        raise ValueError("A fly must have anatomical mesh collision geometry")
    return tuple(names)
