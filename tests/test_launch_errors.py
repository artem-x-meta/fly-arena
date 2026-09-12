"""Launch-time errors a user meets before any long run starts."""
import pytest


def test_missing_graph_is_reported_before_the_body_is_built(tmp_path):
    from fly_arena.config import load_config
    from fly_arena.ethology import EthologySimulation
    with pytest.raises(FileNotFoundError, match="No prepared connectome"):
        EthologySimulation(load_config(), brain_enabled=True, graph=tmp_path / "absent")


def test_unsupported_required_pathway_is_reported_before_a_long_run():
    from fly_arena.neural_ports import NeuralAdapter, NeuralConfig, UnsupportedPortError
    adapter = NeuralAdapter.__new__(NeuralAdapter)
    adapter.registry = {"pathways": {"grooming": {"status": "unsupported",
                                                  "reason": "no functional support"}}}
    adapter.config = NeuralConfig(exploratory_ports=False)
    adapter.indices = {}
    with pytest.raises(UnsupportedPortError, match="grooming"):
        adapter.require_supported(("grooming",))
