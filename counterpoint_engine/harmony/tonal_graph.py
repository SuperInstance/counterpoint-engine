"""
Tonal Graph — directed graph of tonal centers.

Nodes are pitch classes (0–11, C through B).
Edges are chord transitions weighted by frequency.
Directions encode functional harmonic motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_CLASSES = list(range(12))
"""Chromatic pitch classes 0=C, 1=C#, ..., 11=B."""

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
"""Standard pitch class names."""

CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
"""
Pitch classes ordered by perfect fifths ascending.
Index on this circle is the "tonal distance" from C.
"""


def pitch_name(pc: int) -> str:
    """Return the name of a pitch class."""
    if not 0 <= pc < 12:
        raise ValueError(f"Pitch class must be 0-11, got {pc}")
    return PITCH_NAMES[pc]


def pc_from_name(name: str) -> int:
    """Return pitch class from a name like 'C', 'F#', 'Bb'."""
    name = name.strip()
    # Handle flats
    if "b" in name and name not in ("Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb"):
        name = name.replace("b", "")
        idx = PITCH_NAMES.index(name) if name in PITCH_NAMES else None
        if idx is not None:
            return (idx - 1) % 12
    if name in PITCH_NAMES:
        return PITCH_NAMES.index(name)
    # Common enharmonics with flats
    enharmonic = {
        "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
        "Cb": 11, "Fb": 4,
    }
    if name in enharmonic:
        return enharmonic[name]
    raise ValueError(f"Unknown pitch name: {name}")


def circle_of_fifths_position(pc: int) -> int:
    """
    Return the position of a pitch class on the circle of fifths.
    C=0, G=1, D=2, A=3, E=4, B=5, F#=6, Db=7, Ab=8, Eb=9, Bb=10, F=11.
    """
    if not 0 <= pc < 12:
        raise ValueError(f"Pitch class must be 0-11, got {pc}")
    return CIRCLE_OF_FIFTHS.index(pc)


def semitone_interval(from_pc: int, to_pc: int) -> int:
    """Signed semitone interval from one pitch class to another (shortest path)."""
    diff = (to_pc - from_pc) % 12
    if diff > 6:
        diff -= 12
    return diff


# ---------------------------------------------------------------------------
# Direction enum
# ---------------------------------------------------------------------------

class TransitionDirection(Enum):
    """Functional direction of a chord transition."""
    DOMINANT = auto()      # Up a perfect fifth (or equivalent) — tension
    SUBDOMINANT = auto()   # Down a perfect fifth / up a fourth — relaxation away
    RESOLUTION = auto()    # Down a perfect fifth to tonic — closure
    MEDIANT = auto()       # Up or down a major/minor third — color change
    STEP = auto()          # Major or minor second — smooth voice leading
    TRITONE = auto()       # Diminished fifth/augmented fourth — high tension
    NO_MOTION = auto()     # Same root repeated
    UNKNOWN = auto()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


def classify_direction(from_pc: int, to_pc: int) -> TransitionDirection:
    """
    Classify the functional direction of a root movement.

    Parameters
    ----------
    from_pc, to_pc : int
        Pitch classes 0-11.

    Returns
    -------
    TransitionDirection
    """
    diff = (to_pc - from_pc) % 12
    if diff == 0:
        return TransitionDirection.NO_MOTION
    if diff == 7:
        return TransitionDirection.DOMINANT      # up a fifth
    if diff == 5:
        return TransitionDirection.SUBDOMINANT   # up a fourth (down a fifth)
    if diff == 10 or diff == -2:
        return TransitionDirection.RESOLUTION    # down a major second (V→I)
    if diff == 2:
        return TransitionDirection.STEP          # up a major second
    if diff in (3, 4, 8, 9):
        return TransitionDirection.MEDIANT       # thirds and sixths
    if diff == 6:
        return TransitionDirection.TRITONE
    return TransitionDirection.STEP


# ---------------------------------------------------------------------------
# Edge dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Edge:
    """A directed edge in the tonal graph."""
    source: int
    target: int
    weight: float = 1.0
    direction: TransitionDirection = TransitionDirection.UNKNOWN


# ---------------------------------------------------------------------------
# TonalGraph
# ---------------------------------------------------------------------------

class TonalGraph:
    """
    Directed graph of tonal centers.

    Nodes are pitch classes 0–11.  Edges represent chord transitions
    observed in a corpus or progression, weighted by frequency.
    """

    def __init__(self) -> None:
        self._edges: Dict[Tuple[int, int], Edge] = {}
        self._outgoing: Dict[int, List[int]] = {pc: [] for pc in PITCH_CLASSES}
        self._incoming: Dict[int, List[int]] = {pc: [] for pc in PITCH_CLASSES}
        self._total_weight: float = 0.0

    # -- mutation ----------------------------------------------------------

    def add_transition(
        self,
        from_pc: int,
        to_pc: int,
        weight: float = 1.0,
        direction: Optional[TransitionDirection] = None,
    ) -> None:
        """Add (or increment) a directed transition between two pitch classes."""
        if not (0 <= from_pc < 12 and 0 <= to_pc < 12):
            raise ValueError("Pitch classes must be in 0..11")

        key = (from_pc, to_pc)
        if direction is None:
            direction = classify_direction(from_pc, to_pc)

        if key in self._edges:
            old = self._edges[key]
            new_weight = old.weight + weight
            self._edges[key] = Edge(from_pc, to_pc, new_weight, direction)
            self._total_weight += weight
        else:
            self._edges[key] = Edge(from_pc, to_pc, weight, direction)
            self._outgoing[from_pc].append(to_pc)
            self._incoming[to_pc].append(from_pc)
            self._total_weight += weight

    def build_from_progression(self, roots: List[int]) -> None:
        """Populate the graph from a sequence of chord roots."""
        for a, b in zip(roots, roots[1:]):
            self.add_transition(a, b)

    # -- queries -----------------------------------------------------------

    def get_edge(self, from_pc: int, to_pc: int) -> Optional[Edge]:
        """Return the edge between two pitch classes, if it exists."""
        return self._edges.get((from_pc, to_pc))

    def get_direction(self, from_pc: int, to_pc: int) -> TransitionDirection:
        """Return the functional direction of a transition."""
        edge = self.get_edge(from_pc, to_pc)
        if edge is not None:
            return edge.direction
        return classify_direction(from_pc, to_pc)

    def out_degree(self, pc: int) -> int:
        """Number of distinct outgoing transitions."""
        return len(self._outgoing[pc])

    def in_degree(self, pc: int) -> int:
        """Number of distinct incoming transitions."""
        return len(self._incoming[pc])

    def weight(self, from_pc: int, to_pc: int) -> float:
        """Weight of a transition (0.0 if absent)."""
        edge = self.get_edge(from_pc, to_pc)
        return edge.weight if edge else 0.0

    def normalized_weight(self, from_pc: int, to_pc: int) -> float:
        """Transition weight divided by total graph weight."""
        if self._total_weight == 0:
            return 0.0
        return self.weight(from_pc, to_pc) / self._total_weight

    def transition_probability(self, from_pc: int, to_pc: int) -> float:
        """Probability of choosing *to_pc* given we are at *from_pc*."""
        out = self._outgoing[from_pc]
        if not out:
            return 0.0
        total = sum(self.weight(from_pc, t) for t in out)
        return self.weight(from_pc, to_pc) / total if total else 0.0

    def adjacency_matrix(self) -> List[List[float]]:
        """Return a 12×12 adjacency matrix of normalized weights."""
        return [[self.normalized_weight(i, j) for j in PITCH_CLASSES]
                for i in PITCH_CLASSES]

    def neighbors(self, pc: int) -> List[int]:
        """Outgoing neighbors of a pitch class."""
        return list(self._outgoing[pc])

    def __repr__(self) -> str:
        return (
            f"<TonalGraph nodes=12 edges={len(self._edges)} "
            f"total_weight={self._total_weight:.1f}>"
        )
