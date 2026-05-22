"""
Species counterpoint as FLUX constraints.

Each function takes voice data and returns SAT or UNSAT.
A voice is a sequence of pitch classes (0-11) or MIDI note numbers.
A beat is an integer index into the voice sequence.

The rules encode the classical species counterpoint constraints:
- No parallel perfect intervals (fifths, octaves)
- Proper resolution of leading tone to tonic
- Maximum melodic leap of a minor seventh (10 semitones)
- Consonant intervals at strong beats
- Voice independence via Laman rigidity
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from constraint_theory_core.rigidity import is_laman

# ---------------------------------------------------------------------------
# FLUX result literals
# ---------------------------------------------------------------------------

SAT: str = "SAT"
UNSAT: str = "UNSAT"

# ---------------------------------------------------------------------------
# Musical constants
# ---------------------------------------------------------------------------

PERFECT_FIFTH: int = 7      # semitones between pitch classes
PERFECT_OCTAVE: int = 12    # semitones
MAJOR_SIXTH: int = 9
MINOR_SIXTH: int = 8
MAJOR_THIRD: int = 4
MINOR_THIRD: int = 3
MAJOR_SECOND: int = 2
MINOR_SECOND: int = 1
MINOR_SEVENTH_LEAP: int = 10  # maximum allowed melodic leap

# Consonant intervals (in semitones, modulo octave)
CONSONANT_INTERVALS: Tuple[int, ...] = (
    0,   # unison
    3,   # minor third
    4,   # major third
    7,   # perfect fifth
    8,   # minor sixth
    9,   # major sixth
    12,  # perfect octave
)

# Dissonant intervals that must resolve (in species counterpoint)
DISSONANT_INTERVALS: Tuple[int, ...] = (
    1,   # minor second
    2,   # major second
    6,   # tritone / augmented fourth
    10,  # minor seventh
    11,  # major seventh
)

LEADING_TONE: int = 11  # pitch class of leading tone (B in C major)
TONIC: int = 0          # pitch class of tonic (C in C major)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pitch_class(pitch: int) -> int:
    """Return pitch class 0-11 from a MIDI note or pitch class."""
    return pitch % 12


def _interval_at(voice_a: Sequence[int], voice_b: Sequence[int], beat: int) -> int:
    """Absolute interval in semitones between two voices at a beat."""
    return abs(voice_a[beat] - voice_b[beat])


def _interval_class_at(voice_a: Sequence[int], voice_b: Sequence[int], beat: int) -> int:
    """Interval class (mod 12) between two voices at a beat."""
    return abs(_pitch_class(voice_a[beat]) - _pitch_class(voice_b[beat]))


def _motion_type(
    voice_a: Sequence[int], voice_b: Sequence[int], beat: int
) -> str:
    """Return motion type between beat-1 and beat: 'similar', 'contrary', 'oblique', 'static'."""
    if beat == 0:
        return "static"
    da = voice_a[beat] - voice_a[beat - 1]
    db = voice_b[beat] - voice_b[beat - 1]
    if da == 0 and db == 0:
        return "static"
    if da == 0 or db == 0:
        return "oblique"
    if (da > 0 and db > 0) or (da < 0 and db < 0):
        return "similar"
    return "contrary"


# ---------------------------------------------------------------------------
# FLUX constraint kernels
# ---------------------------------------------------------------------------

def no_parallel_fifths(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beats: Sequence[int],
) -> str:
    """Check that no two consecutive beats contain parallel perfect fifths.

    In strict counterpoint, parallel fifths occur when two voices move
    in similar motion to maintain a perfect fifth interval. We check
    consecutive beats in the provided beat list.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences (MIDI notes or pitch classes).
    beats : Sequence[int]
        Beat indices to check. Consecutive indices in this list are
        checked against each other.

    Returns
    -------
    SAT or UNSAT
    """
    if len(beats) < 2:
        return SAT

    for i in range(1, len(beats)):
        prev = beats[i - 1]
        curr = beats[i]
        int_prev = _interval_class_at(voice_a, voice_b, prev)
        int_curr = _interval_class_at(voice_a, voice_b, curr)
        if int_prev == PERFECT_FIFTH and int_curr == PERFECT_FIFTH:
            motion = _motion_type(voice_a, voice_b, curr)
            if motion in ("similar", "static"):
                return UNSAT
    return SAT


def no_parallel_octaves(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beats: Sequence[int],
) -> str:
    """Check that no two consecutive beats contain parallel perfect octaves.

    Parallel octaves destroy voice independence — the voices collapse
    into a single perceived line.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beats : Sequence[int]
        Beat indices to check.

    Returns
    -------
    SAT or UNSAT
    """
    if len(beats) < 2:
        return SAT

    for i in range(1, len(beats)):
        prev = beats[i - 1]
        curr = beats[i]
        int_prev = _interval_at(voice_a, voice_b, prev) % PERFECT_OCTAVE
        int_curr = _interval_at(voice_a, voice_b, curr) % PERFECT_OCTAVE
        if int_prev == 0 and int_curr == 0:
            motion = _motion_type(voice_a, voice_b, curr)
            if motion in ("similar", "static"):
                return UNSAT
    return SAT


def proper_resolution(
    voice: Sequence[int],
    beat: int,
    key_tonic: int = TONIC,
    key_leading: int = LEADING_TONE,
) -> str:
    """Check that a leading tone resolves to the tonic at the given beat.

    In tonal counterpoint, the leading tone (scale degree 7) must
    resolve upward by semitone to the tonic (scale degree 1) when
    it appears at a weak beat preceding a strong beat.

    We check: if voice[beat-1] is the leading tone, then voice[beat]
    must be the tonic (or its octave).

    Parameters
    ----------
    voice : Sequence[int]
        Single pitch sequence.
    beat : int
        Beat index to check resolution *into* (i.e., beat-1 → beat).
    key_tonic : int, default 0
        Pitch class of the tonic.
    key_leading : int, default 11
        Pitch class of the leading tone.

    Returns
    -------
    SAT or UNSAT
    """
    if beat == 0:
        return SAT
    prev_pc = _pitch_class(voice[beat - 1])
    curr_pc = _pitch_class(voice[beat])
    if prev_pc == key_leading and curr_pc != key_tonic:
        return UNSAT
    return SAT


def max_leap_seventh(
    voice: Sequence[int],
    beat: int,
    max_leap: int = MINOR_SEVENTH_LEAP,
) -> str:
    """Check that the melodic leap at beat does not exceed max_leap semitones.

    In singable counterpoint, leaps larger than a minor seventh
    (10 semitones) are forbidden. Compound intervals are reduced
    to their simple form before checking.

    Parameters
    ----------
    voice : Sequence[int]
        Single pitch sequence.
    beat : int
        Beat index to check leap *at* (i.e., voice[beat-1] → voice[beat]).
    max_leap : int, default 10
        Maximum allowed leap in semitones.

    Returns
    -------
    SAT or UNSAT
    """
    if beat == 0:
        return SAT
    leap = abs(voice[beat] - voice[beat - 1])
    # Reduce compound intervals to simple (mod 12, but keep direction info)
    # Actually for leaps we care about absolute distance regardless of octave
    # But counterpoint rules usually apply to simple intervals.
    # We'll reduce to within an octave for the check.
    simple_leap = leap % PERFECT_OCTAVE
    if simple_leap > max_leap and leap > max_leap:
        # Allow octave leaps (12) even though 12 > 10
        if simple_leap != PERFECT_OCTAVE:
            return UNSAT
    return SAT


def consonant_interval(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beat: int,
    allowed: Tuple[int, ...] = CONSONANT_INTERVALS,
) -> str:
    """Check that the interval between two voices at beat is consonant.

    In first-species counterpoint, only perfect and imperfect consonances
    are allowed on strong beats.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Beat index to check.
    allowed : tuple of int
        Allowed interval classes in semitones.

    Returns
    -------
    SAT or UNSAT
    """
    int_class = _interval_class_at(voice_a, voice_b, beat)
    if int_class not in allowed:
        return UNSAT
    return SAT


def voice_independence(laman_check: bool) -> str:
    """Check that the constraint graph is minimally rigid (Laman).

    A set of N voices is independent iff the constraint graph on
    those voices has exactly 2N - 3 edges and satisfies the Laman
    subset condition. This guarantees no voice is redundant and
    every constraint is load-bearing.

    Parameters
    ----------
    laman_check : bool
        Result of is_laman(n_voices, edges).

    Returns
    -------
    SAT or UNSAT
    """
    return SAT if laman_check else UNSAT
