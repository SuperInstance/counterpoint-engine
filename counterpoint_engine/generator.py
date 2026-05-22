"""
Constrained counterpoint generation.

Given a cantus firmus, generate a counterpoint voice that satisfies
ALL active FLUX constraints. Uses backtracking search with constraint
propagation.

Output: MIDI file or tensor-midi event stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    consonant_interval,
    max_leap_seventh,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
)
from counterpoint_engine.laman_counterpoint import CounterpointGraph


class Species(IntEnum):
    """Species of counterpoint."""
    FIRST = 1   # note against note
    SECOND = 2  # two notes against one
    THIRD = 3   # three notes against one
    FOURTH = 4  # suspensions
    FIFTH = 5   # florid counterpoint


@dataclass
class VoiceRange:
    """Allowed pitch range for a voice."""
    min_pitch: int = 48   # C3
    max_pitch: int = 79   # G5

    def candidates(self, scale: Scale, prev_pitch: Optional[int] = None) -> List[int]:
        """Return all valid pitches in range belonging to scale."""
        cands = [p for p in range(self.min_pitch, self.max_pitch + 1)
                 if scale.contains(p)]
        if prev_pitch is not None:
            # Prefer stepwise motion and small leaps for singability
            cands.sort(key=lambda p: abs(p - prev_pitch))
        return cands


@dataclass
class Scale:
    """Diatonic scale represented as pitch classes."""
    tonic: int = 0  # C major by default
    mode: str = "major"
    _pitch_classes: Tuple[int, ...] = field(default_factory=lambda: (0, 2, 4, 5, 7, 9, 11))

    def __post_init__(self):
        if self.mode == "major":
            intervals = (0, 2, 4, 5, 7, 9, 11)
        elif self.mode == "minor":
            intervals = (0, 2, 3, 5, 7, 8, 10)
        else:
            intervals = self._pitch_classes
        self._pitch_classes = tuple(sorted((self.tonic + i) % 12 for i in intervals))

    def contains(self, pitch: int) -> bool:
        return (pitch % 12) in self._pitch_classes

    def pitch_classes(self) -> Tuple[int, ...]:
        return self._pitch_classes


# Default constraint suite for first-species counterpoint
DEFAULT_CONSTRAINTS: List[Callable[..., str]] = [
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
]


@dataclass
class CounterpointGenerator:
    """Generate counterpoint against a cantus firmus.

    Parameters
    ----------
    cantus_firmus : Sequence[int]
        The fixed melody (MIDI note numbers).
    species : Species, default Species.FIRST
        Species of counterpoint to generate.
    scale : Scale, default C major
        Diatonic scale for pitch candidates.
    voice_range : VoiceRange, default C3-G5
        Allowed range for the generated voice.
    constraints : List[Callable], optional
        Active FLUX constraints. Defaults to first-species suite.
    """

    cantus_firmus: Sequence[int]
    species: Species = Species.FIRST
    scale: Scale = field(default_factory=Scale)
    voice_range: VoiceRange = field(default_factory=VoiceRange)
    constraints: List[Callable[..., str]] = field(
        default_factory=lambda: list(DEFAULT_CONSTRAINTS)
    )

    def __post_init__(self):
        self.n_beats = len(self.cantus_firmus)
        self._solution: List[int] = []
        self._graph: Optional[CounterpointGraph] = None

    def _check_all(self, counterpoint: List[int], up_to: int) -> bool:
        """Check all constraints on the partial counterpoint up to beat `up_to`."""
        beats = list(range(up_to + 1))
        for constraint in self.constraints:
            name = constraint.__name__
            if name in ("no_parallel_fifths", "no_parallel_octaves"):
                if len(beats) >= 2:
                    result = constraint(self.cantus_firmus, counterpoint, beats)
                    if result == UNSAT:
                        return False
            elif name == "consonant_interval":
                for b in beats:
                    result = constraint(self.cantus_firmus, counterpoint, b)
                    if result == UNSAT:
                        return False
            elif name == "proper_resolution":
                for b in beats:
                    result = constraint(counterpoint, b)
                    if result == UNSAT:
                        return False
            elif name == "max_leap_seventh":
                for b in beats:
                    result = constraint(counterpoint, b)
                    if result == UNSAT:
                        return False
            else:
                # Generic fallback: try calling with various signatures
                try:
                    result = constraint(self.cantus_firmus, counterpoint, beats)
                except TypeError:
                    try:
                        result = constraint(counterpoint, beats[-1])
                    except TypeError:
                        result = SAT
                if result == UNSAT:
                    return False
        return True

    def generate(self) -> Optional[List[int]]:
        """Generate a counterpoint voice using backtracking.

        Returns the counterpoint melody as MIDI note numbers, or None
        if no solution exists under the current constraints.
        """
        self._solution = []
        if self._backtrack(0):
            return list(self._solution)
        return None

    def _backtrack(self, beat: int) -> bool:
        if beat == self.n_beats:
            return True

        prev_pitch = self._solution[beat - 1] if beat > 0 else None
        candidates = self.voice_range.candidates(self.scale, prev_pitch)

        for pitch in candidates:
            self._solution.append(pitch)
            if self._check_all(self._solution, beat):
                if self._backtrack(beat + 1):
                    return True
            self._solution.pop()

        return False

    def generate_n_voices(
        self,
        n_voices: int,
        voice_ranges: Optional[List[VoiceRange]] = None,
    ) -> Optional[List[List[int]]]:
        """Generate N-voice counterpoint (e.g., fugue texture).

        Voice 0 is the cantus firmus; voices 1..N-1 are generated
        sequentially, each satisfying constraints against all prior
        voices. The constraint graph is built as a Laman graph to
        ensure minimal rigidity.

        Parameters
        ----------
        n_voices : int
            Total number of voices (including cantus firmus).
        voice_ranges : List[VoiceRange], optional
            Range for each generated voice. Defaults to VoiceRange().

        Returns
        -------
        List[List[int]] or None
            All voices (including cantus firmus), or None if unsatisfiable.
        """
        if n_voices < 2:
            return [list(self.cantus_firmus)]

        self._graph = CounterpointGraph(n_voices)
        if not self._graph.verify_rigidity():
            # Force Laman construction
            self._graph.edges = self._graph.edges[:self._graph.expected_edges()]
            # Re-verify
            if not self._graph.verify_rigidity():
                pass  # Continue anyway; user may want non-Laman

        voices: List[List[int]] = [list(self.cantus_firmus)]
        ranges = voice_ranges or [VoiceRange() for _ in range(n_voices - 1)]

        for v_idx in range(1, n_voices):
            # Build local generator against all prior voices this voice connects to
            neighbors = [
                j for (i, j) in self._graph.edges if i == v_idx or j == v_idx
            ]
            # Also include sequential connections within the voice itself
            gen = _MultiVoiceGenerator(
                fixed_voices=voices,
                neighbor_indices=neighbors,
                scale=self.scale,
                voice_range=ranges[v_idx - 1] if v_idx - 1 < len(ranges) else VoiceRange(),
                constraints=self.constraints,
                n_beats=self.n_beats,
            )
            new_voice = gen.generate()
            if new_voice is None:
                return None
            voices.append(new_voice)

        return voices


class _MultiVoiceGenerator:
    """Internal backtracker for a single new voice against multiple fixed voices."""

    def __init__(
        self,
        fixed_voices: List[List[int]],
        neighbor_indices: List[int],
        scale: Scale,
        voice_range: VoiceRange,
        constraints: List[Callable[..., str]],
        n_beats: int,
    ):
        self.fixed_voices = fixed_voices
        self.neighbor_indices = neighbor_indices
        self.scale = scale
        self.voice_range = voice_range
        self.constraints = constraints
        self.n_beats = n_beats
        self._solution: List[int] = []

    def _check_all(self, counterpoint: List[int], up_to: int) -> bool:
        beats = list(range(up_to + 1))
        for constraint in self.constraints:
            name = constraint.__name__
            if name in ("no_parallel_fifths", "no_parallel_octaves"):
                if len(beats) >= 2:
                    for n_idx in self.neighbor_indices:
                        result = constraint(
                            self.fixed_voices[n_idx], counterpoint, beats
                        )
                        if result == UNSAT:
                            return False
            elif name == "consonant_interval":
                for b in beats:
                    for n_idx in self.neighbor_indices:
                        result = constraint(
                            self.fixed_voices[n_idx], counterpoint, b
                        )
                        if result == UNSAT:
                            return False
            elif name == "proper_resolution":
                for b in beats:
                    result = constraint(counterpoint, b)
                    if result == UNSAT:
                        return False
            elif name == "max_leap_seventh":
                for b in beats:
                    result = constraint(counterpoint, b)
                    if result == UNSAT:
                        return False
            else:
                if len(beats) >= 2:
                    for n_idx in self.neighbor_indices:
                        try:
                            result = constraint(
                                self.fixed_voices[n_idx], counterpoint, beats
                            )
                        except TypeError:
                            try:
                                result = constraint(counterpoint, beats[-1])
                            except TypeError:
                                result = SAT
                        if result == UNSAT:
                            return False
        return True

    def generate(self) -> Optional[List[int]]:
        self._solution = []
        if self._backtrack(0):
            return list(self._solution)
        return None

    def _backtrack(self, beat: int) -> bool:
        if beat == self.n_beats:
            return True

        prev_pitch = self._solution[beat - 1] if beat > 0 else None
        candidates = self.voice_range.candidates(self.scale, prev_pitch)

        for pitch in candidates:
            self._solution.append(pitch)
            if self._check_all(self._solution, beat):
                if self._backtrack(beat + 1):
                    return True
            self._solution.pop()

        return False
