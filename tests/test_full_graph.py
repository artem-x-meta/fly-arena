"""Optional full-size causal check; download the official dataset first."""
import os
from pathlib import Path
import numpy as np
import pytest
from fly_arena.brain import Brain


@pytest.mark.skipif(not os.environ.get("FLY_ARENA_TEST_GRAPH"), reason="Requires the prepared MaleCNS graph")
def test_visual_ablation_reaches_descending_readouts():
    graph = Path(os.environ["FLY_ARENA_TEST_GRAPH"])
    commands = []
    spike_sums = []
    for blind in (False, True):
        brain = Brain(graph, seed=1)
        trace = []
        total = 0
        for _ in range(100):
            command, counts = brain.step(np.full(len(brain.retina), .4, np.float32), blind=blind)
            trace.append(command)
            total += int(counts.sum())
        commands.append(trace)
        spike_sums.append(total)
    difference = np.asarray(commands[0]) - commands[1]
    assert np.sqrt(np.mean(difference ** 2)) > .01
    assert spike_sums[0] != spike_sums[1]
