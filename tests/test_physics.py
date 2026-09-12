import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RUN_PHYSICS_TEST") != "1", reason="Requires an OpenGL context")
def test_real_body_walks_stays_upright_and_sees_the_world():
    from fly_arena.arena import Arena
    arena = Arena(seed=1)
    try:
        start = arena.position
        eyes = arena.eyes()
        assert eyes.shape == (2, 96, 96, 3)
        assert eyes.std() > 5
        for _ in range(30):
            arena.step([.8, .8])
        assert np.linalg.norm(arena.position[:2] - start[:2]) > .1
        assert arena.upright > .7
        assert np.isfinite(arena.sim.mj_data.qpos).all()
    finally:
        arena.close()
