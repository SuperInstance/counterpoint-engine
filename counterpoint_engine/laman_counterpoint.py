"""
Map voices to Laman graph for counterpoint rigidity.

N voices → 2N - 3 edges (minimum rigidity).
Each edge = one contrapuntal constraint between a pair of voices.

The theorem: a counterpoint texture is rigid (all voices independently
constrained) iff its constraint graph is Laman.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from constraint_theory_core.rigidity import henneberg_construct as _core_henneberg_construct, is_laman


@dataclass(slots=True)
class CounterpointGraph:
    """A Laman graph representing contrapuntal constraints among voices.

    Attributes
    ----------
    n_voices : int
        Number of voices (vertices).
    edges : List[Tuple[int, int]]
        Edges of the Laman graph. Each edge (i, j) means voices i and j
        are linked by a contrapuntal constraint.
    constraints : Dict[Tuple[int, int], List[str]]
        Mapping from edge to list of constraint names active on that edge.
    """

    n_voices: int
    edges: List[Tuple[int, int]] = field(default_factory=list)
    constraints: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_voices < 2:
            raise ValueError(
                f"n_voices must be at least 2, got {self.n_voices}"
            )
        if not self.edges and self.n_voices >= 2:
            self.edges = _core_henneberg_construct(self.n_voices)
        if not self.constraints:
            self._assign_default_constraints()

    def _assign_default_constraints(self) -> None:
        """Assign standard counterpoint constraints to edges."""
        standard = [
            "no_parallel_fifths",
            "no_parallel_octaves",
            "proper_resolution",
            "max_leap_seventh",
            "consonant_interval",
        ]
        for idx, edge in enumerate(self.edges):
            name = standard[idx % len(standard)]
            self.constraints.setdefault(edge, []).append(name)

    def add_constraint(self, voice_i: int, voice_j: int, name: str) -> None:
        """Add a named constraint to the edge between two voices.

        Creates the edge if it does not already exist.
        """
        edge = (min(voice_i, voice_j), max(voice_i, voice_j))
        if edge not in self.edges:
            self.edges.append(edge)
        self.constraints.setdefault(edge, []).append(name)

    def verify_rigidity(self) -> bool:
        """Check that the current edge set forms a Laman graph.

        Returns True iff the graph is minimally rigid.
        """
        return is_laman(self.n_voices, self.edges)

    def edge_count(self) -> int:
        """Return |E|. For a Laman graph, should equal 2n - 3."""
        return len(self.edges)

    def expected_edges(self) -> int:
        """Return 2n - 3."""
        return 2 * self.n_voices - 3

    def is_minimally_rigid(self) -> bool:
        """True if |E| == 2n - 3 and Laman condition holds."""
        return self.edge_count() == self.expected_edges() and self.verify_rigidity()

    def voice_pairs(self) -> List[Tuple[int, int]]:
        """Return all voice pairs with constraints."""
        return list(self.edges)

    def __repr__(self) -> str:
        return (
            f"CounterpointGraph(n={self.n_voices}, edges={self.edges}, "
            f"rigid={self.verify_rigidity()})"
        )


def henneberg_construct(n: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Build a Laman graph for N voices via Henneberg type-I construction.

    This is a thin wrapper around constraint_theory_core.rigidity.henneberg_construct
    with counterpoint-specific documentation.

    Parameters
    ----------
    n : int
        Number of voices (must be ≥ 2).
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[int, int]]
        Edges of the constructed Laman graph.
    """
    return _core_henneberg_construct(n, seed=seed)


def verify_rigidity(n_voices: int, edges: List[Tuple[int, int]]) -> bool:
    """Check that a constraint graph on N voices is Laman rigid.

    Parameters
    ----------
    n_voices : int
        Number of voices.
    edges : List[Tuple[int, int]]
        Constraint edges.

    Returns
    -------
    bool
        True if the graph satisfies both Laman conditions.
    """
    return is_laman(n_voices, edges)
