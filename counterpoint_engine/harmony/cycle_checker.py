"""
Cycle Checker — verify holonomy consistency of chord progressions.

A progression has zero holonomy when the product (sum) of its tonal
movements closes exactly — i.e. you return to the same tonal center.

Key insight from constraint theory (Section 3c):
  T → S → D → T  (tonic → subdominant → dominant → tonic)
has zero holonomy because the cycle closes.

A modulation is a holonomy violation — you leave one key and arrive
at a different one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple

from .tonal_graph import (
    circle_of_fifths_position,
    semitone_interval,
    TransitionDirection,
    classify_direction,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ProgressionType(Enum):
    """Classification of a chord progression by holonomy signature."""
    DIATONIC = auto()           # Zero holonomy, stays in one key
    MODAL_INTERCHANGE = auto()  # Brief borrow from parallel mode, small holonomy
    MODULATION = auto()         # Key change, non-zero holonomy
    CHROMATIC_MEDIANT = auto()  # Root motion by third, specific holonomy pattern
    CHROMATIC = auto()          # Highly chromatic, large or erratic holonomy
    UNKNOWN = auto()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


# ---------------------------------------------------------------------------
# HolonomyResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HolonomyResult:
    """
    Result of a holonomy analysis on a chord progression.

    Attributes
    ----------
    holonomy : int
        Net displacement in semitones from the original tonic.
        0 means the progression closes in the same key.
    winding_number : float
        Net rotations around the circle of fifths.
    max_deviation : int
        Maximum absolute displacement reached during the progression.
    progression_type : ProgressionType
        Classification based on holonomy signature.
    steps : List[Tuple[int, int, TransitionDirection]]
        Per-step (from_pc, to_pc, direction) record.
    cumulative : List[int]
        Cumulative tonal displacement after each chord.
    """
    holonomy: int
    winding_number: float
    max_deviation: int
    progression_type: ProgressionType
    steps: List[Tuple[int, int, TransitionDirection]] = field(repr=False)
    cumulative: List[int] = field(repr=False)

    def is_consistent(self) -> bool:
        """True if holonomy is zero (cycle closes exactly)."""
        return self.holonomy == 0

    def __repr__(self) -> str:
        return (
            f"<HolonomyResult holonomy={self.holonomy} "
            f"winding={self.winding_number:.2f} "
            f"max_dev={self.max_deviation} type={self.progression_type.name}>"
        )


# ---------------------------------------------------------------------------
# Core holonomy computation
# ---------------------------------------------------------------------------

def _co5_step(from_pc: int, to_pc: int) -> int:
    """
    Signed step along the circle of fifths from *from_pc* to *to_pc*.
    Positive = clockwise (more sharps), negative = counter-clockwise (more flats).
    """
    p1 = circle_of_fifths_position(from_pc)
    p2 = circle_of_fifths_position(to_pc)
    diff = p2 - p1
    # Prefer shortest path on the circle
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12
    return diff


def compute_holonomy(
    roots: List[int],
    wrap: bool = False,
) -> HolonomyResult:
    """
    Compute the holonomy of a chord-root progression.

    Parameters
    ----------
    roots : List[int]
        Pitch classes of chord roots in order.
    wrap : bool
        If True, treat the progression as a closed cycle by appending
        the first root to the end.  Default False (open path).

    Returns
    -------
    HolonomyResult
    """
    if not roots:
        raise ValueError("roots list is empty")

    path = list(roots)
    if wrap and len(path) > 1:
        path = path + [path[0]]

    steps: List[Tuple[int, int, TransitionDirection]] = []
    cumulative: List[int] = [0]
    total = 0
    max_dev = 0

    for a, b in zip(path, path[1:]):
        direction = classify_direction(a, b)
        # Use circle-of-fifths step as the "direction vector"
        step = _co5_step(a, b)
        total += step
        cumulative.append(total)
        max_dev = max(max_dev, abs(total))
        steps.append((a, b, direction))

    # Winding number: how many full rotations around the circle of fifths
    winding = total / 12.0

    # Convert final circle-of-fifths displacement back to semitones
    # One step on Co5 = 7 semitones (mod 12)
    # So net semitone holonomy = total * 7 mod 12
    holonomy_semitones = (total * 7) % 12
    if holonomy_semitones > 6:
        holonomy_semitones -= 12

    ptype = _classify_from_signature(holonomy_semitones, max_dev, winding, steps)

    return HolonomyResult(
        holonomy=holonomy_semitones,
        winding_number=winding,
        max_deviation=max_dev,
        progression_type=ptype,
        steps=steps,
        cumulative=cumulative,
    )


def _classify_from_signature(
    holonomy: int,
    max_dev: int,
    winding: float,
    steps: List[Tuple[int, int, TransitionDirection]],
) -> ProgressionType:
    """Classify a progression from its holonomy signature."""
    if holonomy == 0 and max_dev <= 2:
        return ProgressionType.DIATONIC

    # Check for chromatic mediants: motion by major or minor third
    third_count = sum(
        1 for a, b, d in steps
        if d == TransitionDirection.MEDIANT and abs(semitone_interval(a, b)) in (3, 4)
    )
    if third_count >= 2 and max_dev >= 3:
        return ProgressionType.CHROMATIC_MEDIANT

    if holonomy == 0 and max_dev > 2:
        # Returned home but wandered — modal interchange or brief tonicization
        return ProgressionType.MODAL_INTERCHANGE

    if abs(winding) >= 0.5 or abs(holonomy) >= 3:
        return ProgressionType.MODULATION

    if max_dev >= 4:
        return ProgressionType.CHROMATIC

    return ProgressionType.UNKNOWN


# ---------------------------------------------------------------------------
# Winding number
# ---------------------------------------------------------------------------

def winding_number(roots: List[int]) -> float:
    """
    Compute the winding number of a progression around the circle of fifths.

    This is the net number of full rotations.  A diatonic I-IV-V-I
    has winding number 0 (it doesn't spiral around the circle).
    Giant Steps has non-zero winding because each major-third jump
    carries you around the circle.

    Parameters
    ----------
    roots : List[int]
        Chord roots in order.

    Returns
    -------
    float
        Net rotations.  Positive = clockwise on circle of fifths.
    """
    if not roots or len(roots) < 2:
        return 0.0
    result = compute_holonomy(roots, wrap=False)
    return result.winding_number


# ---------------------------------------------------------------------------
# Classification wrapper
# ---------------------------------------------------------------------------

def classify_progression(roots: List[int]) -> ProgressionType:
    """
    Classify a chord-root progression by its holonomy signature.

    Parameters
    ----------
    roots : List[int]
        Chord roots in order.

    Returns
    -------
    ProgressionType
    """
    return compute_holonomy(roots).progression_type
