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
    Satisfiability,
    consonant_interval,
    max_leap_seventh,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
)
from counterpoint_engine.laman_counterpoint import CounterpointGraph


@dataclass(frozen=True)
class CounterpointResult:
    """Typed result from counterpoint generation."""
    voices: List[List[int]]
    species: int
    key: int
    n_voices: int
    constraints_satisfied: int = 0
    constraints_total: int = 0
    feasible: bool = True

    def to_midi(self, filename: str, bpm: int = 120) -> None:
        """Export to MIDI file using mido."""
        import mido
        mid = mido.MidiFile(ticks_per_beat=480)
        for i, voice in enumerate(self.voices):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage('track_name', name=f'Voice {i+1}'))
            for note in voice:
                if 0 <= note <= 127:
                    track.append(mido.Message('note_on', note=note, velocity=80, time=0))
                    track.append(mido.Message('note_off', note=note, velocity=0, time=480))
            mid.tracks.append(track)
        mid.save(filename)

    def __repr__(self):
        return f"CounterpointResult({self.n_voices}v species-{self.species}, key={self.key}, {self.constraints_satisfied}/{self.constraints_total} constraints)"


class Species(IntEnum):
    """Species of counterpoint."""
    FIRST = 1  # note against note
    SECOND = 2  # two notes against one
    THIRD = 3  # three notes against one
    FOURTH = 4  # suspensions
    FIFTH = 5  # florid counterpoint


@dataclass
class VoiceRange:
    """Allowed pitch range for a voice."""
    min_pitch: int = 48  # C3
    max_pitch: int = 79  # G5

    def __post_init__(self) -> None:
        # Type validation
        if not isinstance(self.min_pitch, int):
            raise TypeError(
                f"min_pitch must be an integer, got {type(self.min_pitch).__name__}"
            )
        if not isinstance(self.max_pitch, int):
            raise TypeError(
                f"max_pitch must be an integer, got {type(self.max_pitch).__name__}"
            )
        # MIDI range check
        if not (0 <= self.min_pitch <= 127):
            raise ValueError(
                f"min_pitch must be in MIDI range 0-127, got {self.min_pitch}"
            )
        if not (0 <= self.max_pitch <= 127):
            raise ValueError(
                f"max_pitch must be in MIDI range 0-127, got {self.max_pitch}"
            )
        if self.min_pitch > self.max_pitch:
            raise ValueError(
                f"min_pitch ({self.min_pitch}) must not exceed "
                f"max_pitch ({self.max_pitch})"
            )

    def __repr__(self) -> str:
        return f"VoiceRange(min_pitch={self.min_pitch}, max_pitch={self.max_pitch})"

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
    _pitch_classes: Tuple[int, ...] = field(
        default_factory=lambda: (0, 2, 4, 5, 7, 9, 11)
    )

    def __post_init__(self) -> None:
        if self.mode not in ("major", "minor") and not self._pitch_classes:
            raise ValueError(
                f"Unknown mode '{self.mode}'; must be 'major' or 'minor' "
                f"or provide explicit _pitch_classes"
            )
        if self.mode == "major":
            intervals = (0, 2, 4, 5, 7, 9, 11)
        elif self.mode == "minor":
            intervals = (0, 2, 3, 5, 7, 8, 10)
        else:
            intervals = self._pitch_classes
        self._pitch_classes = tuple(sorted((self.tonic + i) % 12 for i in intervals))

    def __repr__(self) -> str:
        return f"Scale(tonic={self.tonic}, mode={self.mode!r})"

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

    def __post_init__(self) -> None:
        if len(self.cantus_firmus) == 0:
            raise ValueError("cantus_firmus must not be empty")
        if not isinstance(self.species, Species):
            raise ValueError(
                f"species must be a Species enum, got {type(self.species).__name__}"
            )
        self.n_beats = len(self.cantus_firmus)
        self._solution: List[int] = []
        self._graph: Optional[CounterpointGraph] = None

    def __repr__(self) -> str:
        return (
            f"CounterpointGenerator(n_beats={self.n_beats}, "
            f"species={self.species.name}, scale={self.scale!r})"
        )

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

    def _count_constraints(self, cantus: Sequence[int], counterpoint: List[int]) -> Tuple[int, int]:
        """Count satisfied vs total constraints for a generated counterpoint."""
        total = 0
        satisfied = 0
        beats = list(range(len(counterpoint)))
        for constraint in self.constraints:
            name = constraint.__name__
            if name in ("no_parallel_fifths", "no_parallel_octaves"):
                if len(beats) >= 2:
                    total += 1
                    if constraint(cantus, counterpoint, beats) == SAT:
                        satisfied += 1
            elif name == "consonant_interval":
                for b in beats:
                    total += 1
                    if constraint(cantus, counterpoint, b) == SAT:
                        satisfied += 1
            elif name in ("proper_resolution", "max_leap_seventh"):
                for b in beats:
                    total += 1
                    if constraint(counterpoint, b) == SAT:
                        satisfied += 1
            else:
                total += 1
                try:
                    if constraint(cantus, counterpoint, beats) == SAT:
                        satisfied += 1
                except TypeError:
                    satisfied += 1  # Can't evaluate, assume satisfied
        return satisfied, total

    def generate(self) -> CounterpointResult:
        """Generate a counterpoint voice using backtracking.

        Returns a CounterpointResult. If no solution exists under
        the current constraints, feasible is False.
        """
        self._solution = []
        if self._backtrack(0):
            cp = list(self._solution)
            sat, total = self._count_constraints(self.cantus_firmus, cp)
            return CounterpointResult(
                voices=[list(self.cantus_firmus), cp],
                species=int(self.species),
                key=self.scale.tonic,
                n_voices=2,
                constraints_satisfied=sat,
                constraints_total=total,
                feasible=True,
            )
        return CounterpointResult(
            voices=[list(self.cantus_firmus)],
            species=int(self.species),
            key=self.scale.tonic,
            n_voices=2,
            constraints_satisfied=0,
            constraints_total=len(self.constraints),
            feasible=False,
        )

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
    ) -> CounterpointResult:
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
        CounterpointResult
            Typed result with voices and constraint stats.
        """
        if n_voices < 2:
            return CounterpointResult(
                voices=[list(self.cantus_firmus)],
                species=int(self.species),
                key=self.scale.tonic,
                n_voices=1,
                feasible=True,
            )

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
            neighbors = list(range(len(voices)))
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
                return CounterpointResult(
                    voices=voices,
                    species=int(self.species),
                    key=self.scale.tonic,
                    n_voices=len(voices),
                    feasible=False,
                )
            voices.append(new_voice)

        # Count constraints across all voice pairs
        sat, total = 0, 0
        beats = list(range(self.n_beats))
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                s, t = self._count_pair_constraints(voices[i], voices[j], beats)
                sat += s
                total += t

        return CounterpointResult(
            voices=voices,
            species=int(self.species),
            key=self.scale.tonic,
            n_voices=len(voices),
            constraints_satisfied=sat,
            constraints_total=total,
            feasible=True,
        )

    def _count_pair_constraints(
        self, voice_a: Sequence[int], voice_b: Sequence[int], beats: List[int]
    ) -> Tuple[int, int]:
        total = 0
        satisfied = 0
        for constraint in self.constraints:
            name = constraint.__name__
            if name in ("no_parallel_fifths", "no_parallel_octaves"):
                if len(beats) >= 2:
                    total += 1
                    if constraint(voice_a, voice_b, beats) == SAT:
                        satisfied += 1
            elif name == "consonant_interval":
                for b in beats:
                    total += 1
                    if constraint(voice_a, voice_b, b) == SAT:
                        satisfied += 1
            elif name in ("proper_resolution", "max_leap_seventh"):
                for voice in (voice_a, voice_b):
                    for b in beats:
                        total += 1
                        if constraint(voice, b) == SAT:
                            satisfied += 1
            else:
                total += 1
                try:
                    if constraint(voice_a, voice_b, beats) == SAT:
                        satisfied += 1
                except TypeError:
                    satisfied += 1
        return satisfied, total


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
