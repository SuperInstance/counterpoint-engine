"""
Counterpoint Bridge — connect holonomy harmony analysis to counterpoint rules.

Provides convenience functions that accept counterpoint voices (lists of MIDI
pitch values) and run holonomy-based tonal analysis on them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .tonal_graph import TonalGraph, TransitionDirection, semitone_interval
from .cycle_checker import (
    compute_holonomy,
    HolonomyResult,
    ProgressionType,
)
from .analyzer import (
    analyze_progression,
    Chord,
    ProgressionAnalysis,
    score_stability as _score_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _voices_to_roots(voices: List[List[int]]) -> List[List[int]]:
    """Extract pitch-class roots from each voice (lowest note per chord)."""
    if not voices:
        return []
    n_steps = len(voices[0])
    roots: List[List[int]] = []
    for voice in voices:
        roots.append([p % 12 for p in voice])
    return roots


def _extract_intervals(voice1: List[int], voice2: List[int]) -> List[int]:
    """Compute signed semitone intervals between corresponding notes of two voices."""
    length = min(len(voice1), len(voice2))
    return [semitone_interval(voice1[i] % 12, voice2[i] % 12) for i in range(length)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_counterpoint_harmony(
    voices: List[List[int]],
    wrap: bool = False,
) -> Dict[str, object]:
    """
    Analyze the harmonic implications of counterpoint voices using holonomy.

    Parameters
    ----------
    voices : List[List[int]]
        Counterpoint voices as returned by
        :class:`~counterpoint_engine.generator.CounterpointGenerator`.
        Each voice is a list of MIDI pitch values.  ``voices[0]`` is assumed
        to be the cantus firmus.
    wrap : bool
        If True, treat the progression as a closed cycle when computing
        holonomy.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"holonomy_by_voice"`` – :class:`HolonomyResult` for each voice
        - ``"combined_roots"`` – pitch-class sequence built from the lowest
          note at each time-step
        - ``"combined_holonomy"`` – :class:`HolonomyResult` for the combined
          root progression
        - ``"stability"`` – float 0–1 stability score for combined roots
        - ``"tonal_graph"`` – :class:`TonalGraph` built from combined roots
    """
    if not voices:
        raise ValueError("voices list is empty")

    # Per-voice holonomy
    holonomy_by_voice: List[HolonomyResult] = []
    for voice in voices:
        pcs = [p % 12 for p in voice]
        holonomy_by_voice.append(compute_holonomy(pcs, wrap=wrap))

    # Combined roots: lowest-sounding pitch class at each step
    n_steps = len(voices[0])
    combined_roots: List[int] = []
    for i in range(n_steps):
        lowest = min(voice[i] for voice in voices if i < len(voice))
        combined_roots.append(lowest % 12)

    combined_holonomy = compute_holonomy(combined_roots, wrap=wrap)

    # Build tonal graph
    graph = TonalGraph()
    graph.build_from_progression(combined_roots)

    # Simple stability heuristic based on combined holonomy
    stability = 1.0
    if combined_holonomy.holonomy != 0:
        stability *= 0.5
    if combined_holonomy.max_deviation > 3:
        stability *= 0.7
    if combined_holonomy.progression_type == ProgressionType.DIATONIC:
        stability = min(1.0, stability + 0.2)
    stability = round(max(0.0, min(1.0, stability)), 3)

    return {
        "holonomy_by_voice": holonomy_by_voice,
        "combined_roots": combined_roots,
        "combined_holonomy": combined_holonomy,
        "stability": stability,
        "tonal_graph": graph,
    }


def check_voice_leading_holonomy(
    voice1: List[int],
    voice2: List[int],
    wrap: bool = False,
) -> Dict[str, object]:
    """
    Check the holonomy of voice-leading between two counterpoint voices.

    This treats each voice as a pitch-class path and computes holonomy
    on both individually and on the intervallic relationship between them.

    Parameters
    ----------
    voice1, voice2 : List[int]
        MIDI pitch sequences for two voices (e.g. cantus firmus and
        a counter-subject).
    wrap : bool
        If True, close the cycle when computing holonomy.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"voice1_holonomy"`` – :class:`HolonomyResult` for voice 1
        - ``"voice2_holonomy"`` – :class:`HolonomyResult` for voice 2
        - ``"interval_holonomy"`` – :class:`HolonomyResult` for the
          interval sequence between the two voices
        - ``"consistent"`` – True if both voices individually close and
          the interval sequence also closes (zero holonomy on all three)
        - ``"intervals"`` – list of signed semitone intervals per step
    """
    if not voice1 or not voice2:
        raise ValueError("Both voices must be non-empty")

    pcs1 = [p % 12 for p in voice1]
    pcs2 = [p % 12 for p in voice2]

    h1 = compute_holonomy(pcs1, wrap=wrap)
    h2 = compute_holonomy(pcs2, wrap=wrap)

    # Interval sequence: treat intervals as pitch-class-like values (0–11)
    intervals = _extract_intervals(voice1, voice2)
    # Shift to positive pitch-class range for holonomy computation
    interval_pcs = [(iv % 12) for iv in intervals]
    h_interval = compute_holonomy(interval_pcs, wrap=wrap)

    consistent = (
        h1.holonomy == 0
        and h2.holonomy == 0
        and h_interval.holonomy == 0
    )

    return {
        "voice1_holonomy": h1,
        "voice2_holonomy": h2,
        "interval_holonomy": h_interval,
        "consistent": consistent,
        "intervals": intervals,
    }
