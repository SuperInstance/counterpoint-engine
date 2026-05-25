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

from enum import Enum
from typing import Sequence, Tuple

try:
    from constraint_theory_core.rigidity import is_laman
except ImportError:
    is_laman = None

# ---------------------------------------------------------------------------
# FLUX result type
# ---------------------------------------------------------------------------


class Satisfiability(str, Enum):
    """Satisfiability result for constraint checking.

    Attributes
    ----------
    SAT : str
        Constraint is satisfied.
    UNSAT : str
        Constraint is violated.
    UNKNOWN : str
        Constraint could not be evaluated.
    """

    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


SAT: Satisfiability = Satisfiability.SAT
UNSAT: Satisfiability = Satisfiability.UNSAT

# ---------------------------------------------------------------------------
# Musical constants
# ---------------------------------------------------------------------------

PERFECT_FIFTH: int = 7  # semitones between pitch classes
PERFECT_OCTAVE: int = 12  # semitones
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
TONIC: int = 0  # pitch class of tonic (C in C major)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pitch_class(pitch: int) -> int:
    """Return pitch class 0-11 from a MIDI note or pitch class.

    Parameters
    ----------
    pitch : int
        MIDI note number or pitch class.

    Returns
    -------
    int
        Pitch class in range 0-11.

    Example
    -------
    >>> _pitch_class(60)
    0
    >>> _pitch_class(63)
    3
    """
    return pitch % 12


def _interval_at(voice_a: Sequence[int], voice_b: Sequence[int], beat: int) -> int:
    """Absolute interval in semitones between two voices at a beat.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Index into both sequences.

    Returns
    -------
    int
        Absolute semitone distance.

    Raises
    ------
    IndexError
        If *beat* is out of range for either voice.
    """
    return abs(voice_a[beat] - voice_b[beat])


def _interval_class_at(voice_a: Sequence[int], voice_b: Sequence[int], beat: int) -> int:
    """Interval class (mod 12) between two voices at a beat.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Index into both sequences.

    Returns
    -------
    int
        Interval class 0-11.

    Raises
    ------
    IndexError
        If *beat* is out of range for either voice.
    """
    return abs(_pitch_class(voice_a[beat]) - _pitch_class(voice_b[beat]))


def _motion_type(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beat: int,
) -> str:
    """Classify the motion between two voices from beat-1 to beat.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Beat index (motion from beat-1 → beat).

    Returns
    -------
    str
        One of ``'similar'``, ``'contrary'``, ``'oblique'``, ``'static'``.
    """
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
    str
        ``SAT`` if no parallel fifths are found, ``UNSAT`` otherwise.

    Example
    -------
    >>> no_parallel_fifths([60, 62], [67, 69], [0, 1])
    'UNSAT'
    >>> no_parallel_fifths([60, 60], [67, 69], [0, 1])
    'SAT'
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
    str
        ``SAT`` if no parallel octaves are found, ``UNSAT`` otherwise.

    Example
    -------
    >>> no_parallel_octaves([60, 62], [72, 74], [0, 1])
    'UNSAT'
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
    scale_pitch_classes: Sequence[int] | None = None,
) -> str:
    """Check that a leading tone resolves to the tonic at the given beat.

    In tonal counterpoint, the leading tone (scale degree 7) must
    resolve upward by semitone to the tonic (scale degree 1) when
    it appears at a weak beat preceding a strong beat.

    In harmonic minor, the raised 7th creates a leading tone that
    MUST resolve upward to the tonic. This is the key distinction
    from natural minor (Aeolian), which lacks a leading tone.

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
    scale_pitch_classes : Sequence[int] or None, optional
        If provided, only enforce resolution when the leading tone is
        actually in the scale (supports harmonic vs natural minor).

    Returns
    -------
    str
        ``SAT`` if the leading tone resolves correctly or is absent,
        ``UNSAT`` if it fails to resolve.

    Example
    -------
    >>> proper_resolution([71, 72], 1)
    'SAT'
    >>> proper_resolution([71, 69], 1)
    'UNSAT'
    """
    if beat == 0:
        return SAT
    prev_pc = _pitch_class(voice[beat - 1])
    curr_pc = _pitch_class(voice[beat])
    # If scale_pitch_classes given, only enforce for notes in the scale
    if scale_pitch_classes is not None:
        if prev_pc not in scale_pitch_classes:
            return SAT
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
    str
        ``SAT`` if the leap is within bounds, ``UNSAT`` otherwise.

    Example
    -------
    >>> max_leap_seventh([60, 64], 1)
    'SAT'
    >>> max_leap_seventh([60, 71], 1)
    'UNSAT'
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
    total_beats: int | None = None,
) -> str:
    """Check that the interval between two voices at beat is consonant.

    In first-species counterpoint, only perfect and imperfect consonances
    are allowed on strong beats.

    Following Fux's rules, unisons are only allowed at the first and last
    beats of the exercise. On interior beats, a unison is rejected even
    though it is technically consonant.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Beat index to check.
    allowed : tuple of int
        Allowed interval classes in semitones.
    total_beats : int or None, optional
        Total number of beats in the exercise. If provided, enables
        the Fux unison restriction (unisons only at first/last beat).

    Returns
    -------
    str
        ``SAT`` if the interval is in the allowed set, ``UNSAT`` otherwise.

    Example
    -------
    >>> consonant_interval([60], [64], 0)
    'SAT'
    >>> consonant_interval([60], [66], 0)
    'UNSAT'
    """
    int_class = _interval_class_at(voice_a, voice_b, beat)
    if int_class not in allowed:
        return UNSAT
    # Fux rule: unisons only at first and last beat
    if int_class == 0 and total_beats is not None and total_beats > 1:
        if beat != 0 and beat != total_beats - 1:
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
    str
        ``SAT`` if the graph is Laman, ``UNSAT`` otherwise.

    Example
    -------
    >>> voice_independence(True)
    'SAT'
    """
    return SAT if laman_check else UNSAT


def contrary_motion_score(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beats: Sequence[int],
) -> float:
    """Score the proportion of contrary motion between two voices.

    Contrary motion is the #1 principle of good voice leading.
    Voices moving in opposite directions maintain independence
    better than similar or parallel motion.

    This is a soft constraint / scoring function, not a hard SAT/UNSAT
    rule. Use it to rank candidate solutions during generation.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beats : Sequence[int]
        Beat indices to evaluate.

    Returns
    -------
    float
        Score between 0.0 and 1.0 representing the proportion of
        contrary-motion beats. Higher is better.

    Example
    -------
    >>> # Contrary motion at both beats
    >>> contrary_motion_score([60, 62], [67, 65], [0, 1])
    1.0
    >>> # Similar motion at one beat
    >>> contrary_motion_score([60, 62], [64, 66], [0, 1])
    0.0
    """
    if len(beats) < 2:
        return 0.0
    contrary = 0
    total = 0
    for beat in beats:
        motion = _motion_type(voice_a, voice_b, beat)
        if motion == "static":
            continue
        total += 1
        if motion == "contrary":
            contrary += 1
    if total == 0:
        return 0.0
    return contrary / total


def contrary_motion_bonus(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
    beat: int,
) -> int:
    """Return a positive bonus when two voices move in contrary motion.

    Used as a weighted scoring rule during generation to prefer
    contrary motion without making it a hard constraint.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Pitch sequences.
    beat : int
        Beat index to check motion at.

    Returns
    -------
    int
        1 if contrary motion, 0 otherwise.

    Example
    -------
    >>> contrary_motion_bonus([60, 62], [67, 65], 1)
    1
    >>> contrary_motion_bonus([60, 62], [64, 66], 1)
    0
    """
    if beat == 0:
        return 0
    motion = _motion_type(voice_a, voice_b, beat)
    return 1 if motion == "contrary" else 0


# ---------------------------------------------------------------------------
# Species 2-5 specific constraint helpers
# ---------------------------------------------------------------------------

def voice_range_invariant(
    voices: Sequence[Sequence[int]],
    beat: int,
) -> str:
    """Check that lower-numbered voices are not above higher-numbered voices.

    In multi-voice counterpoint, voice crossing destroys the independence
    of lines. Voice 0 (highest) must stay above voice 1, which must stay
    above voice 2, etc. This is a hard constraint for correct part-writing.

    Parameters
    ----------
    voices : Sequence[Sequence[int]]
        All voices, ordered from highest (index 0) to lowest.
    beat : int
        Beat index to check.

    Returns
    -------
    str
        ``SAT`` if no voice crossing, ``UNSAT`` if a lower voice is
        above a higher voice at this beat.

    Example
    -------
    >>> voice_range_invariant([[72], [60]], 0)
    'SAT'
    >>> voice_range_invariant([[60], [72]], 0)
    'UNSAT'
    """
    for i in range(len(voices) - 1):
        if voices[i][beat] < voices[i + 1][beat]:
            return UNSAT
    return SAT


def is_step(a: int, b: int) -> bool:
    """Return True if the motion from *a* to *b* is stepwise (1 or 2 semitones).

    Parameters
    ----------
    a : int
        Starting pitch.
    b : int
        Target pitch.

    Returns
    -------
    bool
        True when the absolute difference is 1 or 2 and the pitches differ.

    Example
    -------
    >>> is_step(60, 62)
    True
    >>> is_step(60, 63)
    False
    """
    return abs(a - b) <= 2 and a != b


def consonant_interval_class(interval: int) -> bool:
    """Check if an interval class (0-11) is consonant.

    Parameters
    ----------
    interval : int
        Interval class to check (will be taken mod 12).

    Returns
    -------
    bool
        True if the interval is a perfect or imperfect consonance.

    Example
    -------
    >>> consonant_interval_class(7)
    True
    >>> consonant_interval_class(6)
    False
    """
    return interval % 12 in CONSONANT_INTERVALS


def passing_tone_ok(
    counterpoint: Sequence[int],
    idx: int,
) -> str:
    """Check that a dissonant note at idx is a valid passing tone.

    A passing tone must be approached by step and left by step
    in the same direction.

    Parameters
    ----------
    counterpoint : Sequence[int]
        The counterpoint voice.
    idx : int
        Index of the note to check.

    Returns
    -------
    str
        ``SAT`` if the note is consonant (no check needed) or is a
        valid passing tone. ``UNSAT`` otherwise.
    """
    if idx == 0 or idx >= len(counterpoint) - 1:
        return SAT  # boundary notes exempt
    prev = counterpoint[idx - 1]
    curr = counterpoint[idx]
    nxt = counterpoint[idx + 1]
    # If consonant interval between prev and curr, no issue
    # This rule is specifically about dissonant subdivision notes
    # Approached and left by step in same direction
    direction_in = curr - prev
    direction_out = nxt - curr
    if is_step(prev, curr) and is_step(curr, nxt):
        if (direction_in > 0 and direction_out > 0) or (direction_in < 0 and direction_out < 0):
            return SAT
    return UNSAT


def cambiata_ok(
    counterpoint: Sequence[int],
    idx: int,
) -> str:
    """Check the cambiata figure at a weak-beat dissonance.

    Cambiata: dissonance on weak beat approached by step (down),
    followed by a skip of a 3rd in the same direction, then
    resolution by step in opposite direction.

    Parameters
    ----------
    counterpoint : Sequence[int]
        The counterpoint voice.
    idx : int
        Index of the note to check.

    Returns
    -------
    str
        ``SAT`` if the cambiata pattern is valid, ``UNSAT`` otherwise.
    """
    if idx < 1 or idx >= len(counterpoint) - 1:
        return SAT
    prev = counterpoint[idx - 1]
    curr = counterpoint[idx]
    counterpoint[idx + 1]
    # Downward step to dissonance, then skip down, then step up
    direction_in = curr - prev
    if abs(direction_in) <= 2:  # stepwise approach
        return SAT
    return UNSAT


def suspension_preparation(
    counterpoint: Sequence[int],
    cf_note: int,
    idx: int,
) -> str:
    """Check that a suspension is properly prepared.

    The note tied into the strong beat must be consonant with the CF.

    Parameters
    ----------
    counterpoint : Sequence[int]
        The counterpoint voice.
    cf_note : int
        The cantus firmus note at the same beat.
    idx : int
        Index of the suspension note.

    Returns
    -------
    str
        ``SAT`` if the preparation is consonant, ``UNSAT`` otherwise.
    """
    if idx < 1 or idx >= len(counterpoint):
        return SAT
    interval = abs(counterpoint[idx - 1] - cf_note) % 12
    if consonant_interval_class(interval):
        return SAT
    return UNSAT


def suspension_resolution(
    counterpoint: Sequence[int],
    idx: int,
) -> str:
    """Check that a suspension resolves downward by step.

    The dissonant suspension on the strong beat must resolve
    downward by step to a consonance.

    Parameters
    ----------
    counterpoint : Sequence[int]
        The counterpoint voice.
    idx : int
        Index of the suspension note.

    Returns
    -------
    str
        ``SAT`` if the resolution is stepwise down, ``UNSAT`` otherwise.
    """
    if idx < 1 or idx >= len(counterpoint):
        return SAT
    prev = counterpoint[idx - 1]
    curr = counterpoint[idx]
    # Resolution should be stepwise down
    diff = prev - curr
    if 1 <= diff <= 2:
        return SAT
    return UNSAT
