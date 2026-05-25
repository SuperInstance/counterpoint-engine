"""Tests for counterpoint_engine.laman_counterpoint — rigidity mapping."""

import pytest

from counterpoint_engine.laman_counterpoint import (
    CounterpointGraph,
    henneberg_construct,
    verify_rigidity,
)
try:
    from constraint_theory_core.rigidity import is_laman
except ImportError:
    is_laman = None
    is_laman = None


class TestCounterpointGraph:
    def test_two_voices(self):
        g = CounterpointGraph(2)
        assert g.n_voices == 2
        assert g.edge_count() == 1
        assert g.expected_edges() == 1
        assert g.verify_rigidity()
        assert g.is_minimally_rigid()

    def test_three_voices(self):
        g = CounterpointGraph(3)
        assert g.n_voices == 3
        assert g.edge_count() == 3
        assert g.expected_edges() == 3
        assert g.verify_rigidity()

    def test_four_voices(self):
        g = CounterpointGraph(4)
        assert g.n_voices == 4
        assert g.edge_count() == 5
        assert g.expected_edges() == 5
        assert g.verify_rigidity()
        assert g.is_minimally_rigid()

    def test_default_constraints_assigned(self):
        g = CounterpointGraph(4)
        assert len(g.constraints) == 5
        for edge, names in g.constraints.items():
            assert len(names) >= 1
            assert all(isinstance(n, str) for n in names)

    def test_add_constraint(self):
        g = CounterpointGraph(3)
        g.add_constraint(0, 2, "custom_rule")
        assert (0, 2) in g.edges
        assert "custom_rule" in g.constraints[(0, 2)]

    def test_voices_map_to_vertices(self):
        """Voices are vertices; each edge is a contrapuntal constraint."""
        for n in range(2, 12):
            g = CounterpointGraph(n)
            assert g.n_voices == n
            assert g.edge_count() == 2 * n - 3
            assert g.verify_rigidity()


class TestHennebergWrapper:
    def test_matches_core(self):
        try:
            from constraint_theory_core.rigidity import henneberg_construct as core_hb
        except ImportError:
            pytest.skip("constraint-theory-core not installed")
        edges = henneberg_construct(8)
        assert edges == core_hb(8, seed=42)

    def test_reproducible(self):
        e1 = henneberg_construct(10, seed=99)
        e2 = henneberg_construct(10, seed=99)
        assert e1 == e2


class TestVerifyRigidity:
    def test_k3(self):
        assert verify_rigidity(3, [(0, 1), (1, 2), (0, 2)])

    def test_too_few_edges(self):
        assert not verify_rigidity(4, [(0, 1), (1, 2)])

    def test_too_many_edges(self):
        assert not verify_rigidity(4, [
            (0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)
        ])
