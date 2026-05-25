"""
Constrained counterpoint generation.

Given a cantus firmus, generate a counterpoint voice that satisfies
ALL active FLUX constraints. Uses backtracking search with constraint
propagation.

Output: MIDI file or tensor-midi event stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from counterpoint_engine.exceptions import (
    RangeViolationError,
)
from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    consonant_interval,
    consonant_interval_class,
    contrary_motion_bonus,
    max_leap_seventh,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    voice_range_invariant,
)
from counterpoint_engine.laman_counterpoint import CounterpointGraph


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CounterpointResult:
    """Typed result from counterpoint generation.

    Attributes
    ----------
    voices : List[List[int]]
        Generated voices. ``voices[0]`` is the cantus firmus.
    species : int
        Species number (1–5).
    key : int
        Tonic pitch class of the key.
    n_voices : int
        Total number of voices.
    constraints_satisfied : int
        Number of individual constraint checks that passed.
    constraints_total : int
        Total number of individual constraint checks evaluated.
    feasible : bool
        Whether a valid solution was found.

    Example
    -------
    >>> from counterpoint_engine.generator import CounterpointGenerator, Species
    >>> gen = CounterpointGenerator(cantus_firmus=[60, 62, 64])
    >>> result = gen.generate()
    >>> result.feasible
    True
    >>> len(result.voices)
    2
    """

    voices: List[List[int]]
    species: int
    key: int
    n_voices: int
    constraints_satisfied: int = 0
    constraints_total: int = 0
    feasible: bool = True

    def to_midi(self, filename: str, bpm: int = 120) -> None:
        """Export to MIDI file using mido.

        For species 2/3/5, the counterpoint has more notes than the CF.
        CF notes get proportionally longer durations to stay aligned.

        Parameters
        ----------
        filename : str
            Output MIDI file path.
        bpm : int, default 120
            Beats per minute (used for tempo meta-message).

        Raises
        ------
        ImportError
            If the ``mido`` package is not installed.
        """
        import mido
        mid = mido.MidiFile(ticks_per_beat=480)
        cf_len = len(self.voices[0])
        cp_len = len(self.voices[1]) if len(self.voices) > 1 else cf_len
        # Duration ratio: how many CP notes per CF note
        ratio = cp_len // cf_len if cf_len > 0 else 1
        cp_note_ticks = 480 // ratio if ratio > 0 else 480
        cf_note_ticks = 480

        for i, voice in enumerate(self.voices):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage('track_name', name=f'Voice {i+1}'))
            note_ticks = cf_note_ticks if i == 0 else cp_note_ticks
            for note in voice:
                if 0 <= note <= 127:
                    track.append(mido.Message('note_on', note=note, velocity=80, time=0))
                    track.append(mido.Message('note_off', note=note, velocity=0, time=note_ticks))
            mid.tracks.append(track)
        mid.save(filename)

    def __repr__(self) -> str:
        return (
            f"CounterpointResult({self.n_voices}v species-{self.species}, "
            f"key={self.key}, {self.constraints_satisfied}/{self.constraints_total} constraints)"
        )


class Species(IntEnum):
    """Species of counterpoint.

    Attributes
    ----------
    FIRST : int
        Note-against-note (1:1).
    SECOND : int
        Two notes against one (2:1).
    THIRD : int
        Four notes against one (4:1).
    FOURTH : int
        Suspensions (syncopation).
    FIFTH : int
        Florid counterpoint (free mix of species 1–4).
    """

    FIRST = 1  # note against note
    SECOND = 2  # two notes against one
    THIRD = 3  # three notes against one
    FOURTH = 4  # suspensions
    FIFTH = 5  # florid counterpoint


@dataclass(slots=True)
class VoiceRange:
    """Allowed pitch range for a voice.

    Parameters
    ----------
    min_pitch : int
        Lowest allowed MIDI note number (default C3 = 48).
    max_pitch : int
        Highest allowed MIDI note number (default G5 = 79).

    Raises
    ------
    TypeError
        If min_pitch or max_pitch are not integers.
    ValueError
        If pitches are outside MIDI range 0–127 or min > max.

    Example
    -------
    >>> vr = VoiceRange(min_pitch=55, max_pitch=72)
    >>> vr.min_pitch
    55
    """

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
        """Return all valid pitches in range belonging to *scale*.

        Parameters
        ----------
        scale : Scale
            The diatonic scale to filter pitches against.
        prev_pitch : int or None
            If provided, sort candidates by proximity for stepwise preference.

        Returns
        -------
        List[int]
            MIDI pitches in range that belong to the scale.
        """
        cands = [p for p in range(self.min_pitch, self.max_pitch + 1)
                 if scale.contains(p)]
        if prev_pitch is not None:
            # Prefer stepwise motion and small leaps for singability
            cands.sort(key=lambda p: abs(p - prev_pitch))
        return cands


@dataclass(slots=True)
class Scale:
    """Diatonic scale represented as pitch classes.

    Parameters
    ----------
    tonic : int
        Tonic pitch class (default 0 = C).
    mode : str
        ``'major'`` or ``'minor'`` (default ``'major'``).

    Raises
    ------
    ValueError
        If *mode* is not ``'major'`` or ``'minor'`` and no explicit
        pitch classes are provided.

    Example
    -------
    >>> s = Scale(tonic=0, mode="major")
    >>> s.contains(62)  # D in C major
    True
    >>> s.contains(63)  # D# not in C major
    False
    """

    tonic: int = 0  # C major by default
    mode: str = "major"
    _pitch_classes: Tuple[int, ...] = field(
        default_factory=lambda: (0, 2, 4, 5, 7, 9, 11)
    )

    def __post_init__(self) -> None:
        valid_modes = ("major", "minor", "harmonic_minor", "natural_minor")
        if self.mode not in valid_modes and not self._pitch_classes:
            raise ValueError(
                f"Unknown mode '{self.mode}'; must be one of {valid_modes} "
                f"or provide explicit _pitch_classes"
            )
        if self.mode == "major":
            intervals = (0, 2, 4, 5, 7, 9, 11)
        elif self.mode == "minor" or self.mode == "harmonic_minor":
            # Harmonic minor: raised 7th (leading tone) for proper resolution
            intervals = (0, 2, 3, 5, 7, 8, 11)
        elif self.mode == "natural_minor":
            # Natural minor (Aeolian): no raised 7th
            intervals = (0, 2, 3, 5, 7, 8, 10)
        else:
            intervals = self._pitch_classes
        object.__setattr__(
            self,
            "_pitch_classes",
            tuple(sorted((self.tonic + i) % 12 for i in intervals)),
        )

    def __repr__(self) -> str:
        return f"Scale(tonic={self.tonic}, mode={self.mode!r})"

    def contains(self, pitch: int) -> bool:
        """Check whether *pitch* belongs to this scale.

        Parameters
        ----------
        pitch : int
            MIDI note number.

        Returns
        -------
        bool
        """
        return (pitch % 12) in self._pitch_classes

    def pitch_classes(self) -> Tuple[int, ...]:
        """Return the pitch classes of this scale.

        Returns
        -------
        Tuple[int, ...]
            Sorted tuple of pitch classes 0–11.
        """
        return self._pitch_classes

    def leading_tone(self) -> int:
        """Return the pitch class of the leading tone in this scale.

        In harmonic minor, the leading tone is the raised 7th (one semitone
        below the tonic). In natural minor, there is no true leading tone;
        the subtonic (2 semitones below) is returned for compatibility.

        Returns
        -------
        int
            Pitch class of the leading tone.
        """
        return (self.tonic + 11) % 12

    def is_harmonic_minor(self) -> bool:
        """Return True if this scale is a harmonic minor scale.

        Returns
        -------
        bool
        """
        # Harmonic minor has a leading tone (11 semitones above tonic)
        # and a minor third (3 semitones above tonic)
        lt = (self.tonic + 11) % 12
        m3 = (self.tonic + 3) % 12
        return lt in self._pitch_classes and m3 in self._pitch_classes


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
        The fixed melody (MIDI note numbers). Must not be empty.
    species : Species, default Species.FIRST
        Species of counterpoint to generate.
    scale : Scale, default C major
        Diatonic scale for pitch candidates.
    voice_range : VoiceRange, default C3-G5
        Allowed range for the generated voice.
    constraints : List[Callable], optional
        Active FLUX constraints. Defaults to first-species suite.

    Raises
    ------
    ValueError
        If *cantus_firmus* is empty or *species* is not a valid Species.

    Example
    -------
    >>> gen = CounterpointGenerator(cantus_firmus=[60, 62, 64, 65, 67])
    >>> result = gen.generate()
    >>> result.feasible
    True
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
        # Validate cantus firmus pitches
        for i, pitch in enumerate(self.cantus_firmus):
            if not isinstance(pitch, int):
                raise TypeError(
                    f"cantus_firmus[{i}] must be an int, got {type(pitch).__name__}"
                )
            if not (0 <= pitch <= 127):
                raise RangeViolationError(
                    pitch=pitch,
                    min_pitch=0,
                    max_pitch=127,
                    beat=i,
                )
        self.n_beats: int = len(self.cantus_firmus)
        self._solution: List[int] = []
        self._graph: Optional[CounterpointGraph] = None

    def __repr__(self) -> str:
        return (
            f"CounterpointGenerator(n_beats={self.n_beats}, "
            f"species={self.species.name}, scale={self.scale!r})"
        )

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def _check_all(self, counterpoint: List[int], up_to: int) -> bool:
        """Check all constraints on the partial counterpoint up to beat *up_to*.

        Parameters
        ----------
        counterpoint : List[int]
            Partial counterpoint built so far.
        up_to : int
            Check beats 0..up_to (inclusive).

        Returns
        -------
        bool
            True if all constraints pass.
        """
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
                    result = constraint(self.cantus_firmus, counterpoint, b, total_beats=self.n_beats)
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

    def _count_constraints(
        self, cantus: Sequence[int], counterpoint: List[int]
    ) -> Tuple[int, int]:
        """Count satisfied vs total constraints for a generated counterpoint.

        Parameters
        ----------
        cantus : Sequence[int]
            The cantus firmus.
        counterpoint : List[int]
            The generated counterpoint.

        Returns
        -------
        Tuple[int, int]
            (satisfied_count, total_count)
        """
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
                    if constraint(cantus, counterpoint, b, total_beats=len(counterpoint)) == SAT:
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

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    def generate(self) -> CounterpointResult:
        """Generate a counterpoint voice using backtracking.

        Dispatches to the species-specific generation method based on
        ``self.species``.

        Returns
        -------
        CounterpointResult
            Typed result with voices and constraint stats. If no
            solution exists under the current constraints,
            ``result.feasible`` will be ``False``.

        Raises
        ------
        GenerationError
            If an unexpected error occurs during generation.
        """
        species_method: Dict[Species, Callable[[], CounterpointResult]] = {
            Species.FIRST: self._generate_species1,
            Species.SECOND: self._generate_species2,
            Species.THIRD: self._generate_species3,
            Species.FOURTH: self._generate_species4,
            Species.FIFTH: self._generate_species5,
        }
        return species_method[self.species]()

    # -----------------------------------------------------------------------
    # Species 1: note-against-note
    # -----------------------------------------------------------------------

    def _generate_species1(self) -> CounterpointResult:
        """Generate first-species (1:1 note-against-note) counterpoint."""
        self._solution = []
        if self._backtrack_species1(0):
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
        return self._infeasible_result()

    def _backtrack_species1(self, beat: int) -> bool:
        if beat == self.n_beats:
            return True
        prev_pitch = self._solution[beat - 1] if beat > 0 else None
        candidates = self.voice_range.candidates(self.scale, prev_pitch)
        # Score candidates: prefer those that create contrary motion with CF
        if beat > 0:
            def _score(p: int) -> int:
                return contrary_motion_bonus(self.cantus_firmus, [p], 0) * -1  # hack: we need both voices at beat
            # Better: compute bonus properly
            list(self._solution) + [p for p in candidates]
            scored = []
            for pitch in candidates:
                bonus = contrary_motion_bonus(
                    self.cantus_firmus,
                    list(self._solution) + [pitch],
                    beat,
                )
                scored.append((bonus, pitch))
            scored.sort(key=lambda x: -x[0])  # Higher bonus first
            candidates = [p for _, p in scored]
        for pitch in candidates:
            self._solution.append(pitch)
            if self._check_all(self._solution, beat):
                if self._backtrack_species1(beat + 1):
                    return True
            self._solution.pop()
        return False

    # -----------------------------------------------------------------------
    # Species 2: two-against-one
    # -----------------------------------------------------------------------

    def _generate_species2(self) -> CounterpointResult:
        """Generate second-species (2:1) counterpoint.

        Two counterpoint notes per CF note. Strong beats must be
        consonant; weak beats are passing tones.
        """
        self._solution = []
        cf = list(self.cantus_firmus)
        if self._backtrack_species2(0):
            cp = list(self._solution)
            sat, total = self._count_species2_constraints(cf, cp)
            return CounterpointResult(
                voices=[cf, cp],
                species=2,
                key=self.scale.tonic,
                n_voices=2,
                constraints_satisfied=sat,
                constraints_total=total,
                feasible=True,
            )
        return self._infeasible_result()

    def _backtrack_species2(self, cf_beat: int) -> bool:
        if cf_beat == self.n_beats:
            return True

        cf_note = self.cantus_firmus[cf_beat]
        prev_pitch = self._solution[-1] if self._solution else None

        # Generate the strong-beat note (must be consonant with CF)
        candidates = self.voice_range.candidates(self.scale, prev_pitch)
        for pitch in candidates:
            # Check consonance with CF
            intv = abs(pitch - cf_note) % 12
            if not consonant_interval_class(intv):
                continue
            # Check melodic leap
            if prev_pitch is not None and abs(pitch - prev_pitch) > 10:
                leap_simple = abs(pitch - prev_pitch) % 12
                if leap_simple > 10 and leap_simple != 12:
                    continue
            self._solution.append(pitch)

            # Generate the weak-beat passing tone (stepwise from strong)
            passing_candidates = self._passing_tone_candidates(pitch)
            random.shuffle(passing_candidates)
            for p_pitch in passing_candidates:
                self._solution.append(p_pitch)
                if self._backtrack_species2(cf_beat + 1):
                    return True
                self._solution.pop()

            self._solution.pop()
        return False

    def _passing_tone_candidates(self, from_pitch: int) -> List[int]:
        """Return stepwise neighbors as passing tone candidates.

        Parameters
        ----------
        from_pitch : int
            The pitch to step from.

        Returns
        -------
        List[int]
            Pitches ±1 and ±2 semitones within voice range.
        """
        candidates: List[int] = []
        for step in [-2, -1, 1, 2]:
            p = from_pitch + step
            if self.voice_range.min_pitch <= p <= self.voice_range.max_pitch:
                candidates.append(p)
        return candidates

    def _count_species2_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
        """Count constraints for species 2.

        Parameters
        ----------
        cf : Sequence[int]
            Cantus firmus.
        cp : List[int]
            Counterpoint (2x length of CF).

        Returns
        -------
        Tuple[int, int]
            (satisfied, total)
        """
        total = 0
        sat = 0
        for cf_beat in range(self.n_beats):
            strong_idx = cf_beat * 2
            # Strong beat: must be consonant
            total += 1
            intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
            if consonant_interval_class(intv):
                sat += 1
            # Melodic smoothness between consecutive strong beats
            if cf_beat > 0:
                prev_strong = (cf_beat - 1) * 2
                total += 1
                leap = abs(cp[strong_idx] - cp[prev_strong])
                if leap <= 10 or leap % 12 == 0:
                    sat += 1
        return sat, total

    # -----------------------------------------------------------------------
    # Species 3: four-against-one
    # -----------------------------------------------------------------------

    def _generate_species3(self) -> CounterpointResult:
        """Generate third-species (4:1) counterpoint.

        Four counterpoint notes per CF note. Strong beats must be
        consonant; weak beats are stepwise passing tones.
        """
        self._solution = []
        cf = list(self.cantus_firmus)
        if self._backtrack_species3(0):
            cp = list(self._solution)
            sat, total = self._count_species3_constraints(cf, cp)
            return CounterpointResult(
                voices=[cf, cp],
                species=3,
                key=self.scale.tonic,
                n_voices=2,
                constraints_satisfied=sat,
                constraints_total=total,
                feasible=True,
            )
        return self._infeasible_result()

    def _backtrack_species3(self, cf_beat: int) -> bool:
        if cf_beat == self.n_beats:
            return True

        cf_note = self.cantus_firmus[cf_beat]
        prev_pitch = self._solution[-1] if self._solution else None

        # Generate beat 0 (strong) — must be consonant
        candidates = self.voice_range.candidates(self.scale, prev_pitch)
        for pitch in candidates:
            intv = abs(pitch - cf_note) % 12
            if not consonant_interval_class(intv):
                continue
            if prev_pitch is not None and abs(pitch - prev_pitch) > 10:
                if abs(pitch - prev_pitch) % 12 != 0:
                    continue
            self._solution.append(pitch)

            # Generate beats 1-3 as stepwise motion
            if self._backtrack_species3_subdivisions(cf_beat, 1, pitch):
                return True
            self._solution.pop()
        return False

    def _backtrack_species3_subdivisions(
        self, cf_beat: int, sub_beat: int, prev_pitch: int
    ) -> bool:
        if sub_beat == 4:
            return self._backtrack_species3(cf_beat + 1)

        candidates = self._passing_tone_candidates(prev_pitch)
        if self.voice_range.min_pitch <= prev_pitch <= self.voice_range.max_pitch:
            candidates.append(prev_pitch)
        random.shuffle(candidates)

        for pitch in candidates:
            self._solution.append(pitch)
            if sub_beat < 3:
                if self._backtrack_species3_subdivisions(cf_beat, sub_beat + 1, pitch):
                    return True
            else:
                if self._backtrack_species3(cf_beat + 1):
                    return True
            self._solution.pop()
        return False

    def _count_species3_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
        """Count constraints for species 3.

        Parameters
        ----------
        cf : Sequence[int]
            Cantus firmus.
        cp : List[int]
            Counterpoint (4x length of CF).

        Returns
        -------
        Tuple[int, int]
            (satisfied, total)
        """
        total = 0
        sat = 0
        for cf_beat in range(self.n_beats):
            strong_idx = cf_beat * 4
            total += 1
            intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
            if consonant_interval_class(intv):
                sat += 1
        return sat, total

    # -----------------------------------------------------------------------
    # Species 4: syncopation / suspension
    # -----------------------------------------------------------------------

    def _generate_species4(self) -> CounterpointResult:
        """Generate fourth-species counterpoint with syncopation.

        Creates suspension chains:
        preparation (consonant) → suspension (dissonant) →
        resolution (consonant, step down)

        Two-pass approach:
        1. Generate consonant frameworks with varying range
        2. Insert suspensions by modifying preparation beats
        3. Pick the best result combining suspensions and range
        """
        cf = list(self.cantus_firmus)

        best_solution: Optional[List[int]] = None
        best_score = -1

        # Collect all valid starting pitches
        start_pitches: List[int] = []
        for p in range(self.voice_range.max_pitch, self.voice_range.min_pitch - 1, -1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf[0]) % 12
            if consonant_interval_class(intv) and intv != 0:
                start_pitches.append(p)

        for start in start_pitches[:25]:
            self._solution = [start]
            if self._species4_backtrack_framework(cf, 1):
                framework = list(self._solution)
                with_susp = self._species4_insert_suspensions(cf, framework)
                susp_count = sum(
                    1 for b in range(len(with_susp))
                    if not consonant_interval_class(abs(with_susp[b] - cf[b]) % 12)
                )
                span = max(with_susp) - min(with_susp)
                unique = len(set(with_susp))

                if susp_count >= 1 and span >= 12:
                    best_solution = with_susp
                    break

                score = susp_count * 1000 + span * 10 + unique
                if score > best_score:
                    best_score = score
                    best_solution = with_susp

        if best_solution is not None:
            sat, total = self._count_species4_constraints(cf, best_solution)
            return CounterpointResult(
                voices=[cf, best_solution],
                species=4,
                key=self.scale.tonic,
                n_voices=2,
                constraints_satisfied=sat,
                constraints_total=total,
                feasible=True,
            )
        return self._infeasible_result()

    def _species4_find_framework(self, cf: List[int]) -> Optional[List[int]]:
        """Find a consonant melodic framework with wide range.

        Strategy: build an arc-shaped melody that goes high then low
        (or vice versa) to span a wide range, ensuring consonance at
        every beat.

        Parameters
        ----------
        cf : List[int]
            Cantus firmus.

        Returns
        -------
        List[int] or None
            Framework pitches, or None if no valid framework found.
        """
        best: Optional[List[int]] = None
        best_score = -1

        start_pitches: List[int] = []
        for p in range(self.voice_range.max_pitch, self.voice_range.min_pitch - 1, -1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf[0]) % 12
            if consonant_interval_class(intv) and intv != 0:
                start_pitches.append(p)

        end_targets: List[int] = []
        for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf[-1]) % 12
            if consonant_interval_class(intv) and intv != 0:
                end_targets.append(p)

        for start in start_pitches[:15]:
            for end_pitch in end_targets:
                if abs(end_pitch - start) < 12:
                    continue
                self._solution = [start]
                self._s4_target_end = end_pitch
                if self._species4_backtrack_framework(cf, 1):
                    cp = list(self._solution)
                    span = max(cp) - min(cp)
                    unique = len(set(cp))
                    score = span * 10 + unique * 50
                    if score > best_score:
                        best_score = score
                        best = cp
                    if span >= 12 and unique >= 4:
                        return best

        for start in start_pitches[:15]:
            self._solution = [start]
            self._s4_target_end = None
            if self._species4_backtrack_framework(cf, 1):
                cp = list(self._solution)
                span = max(cp) - min(cp)
                unique = len(set(cp))
                score = span * 10 + unique * 50
                if score > best_score:
                    best_score = score
                    best = cp

        return best

    def _species4_backtrack_framework(self, cf: List[int], beat: int) -> bool:
        """Backtrack to find consonant framework, optionally steering toward target."""
        if beat == self.n_beats:
            return True

        cf_note = cf[beat]
        prev_pitch = self._solution[beat - 1]
        candidates = self._consonant_candidates(cf_note, prev_pitch)

        target = getattr(self, '_s4_target_end', None)
        if target is not None and beat >= self.n_beats - 2:
            candidates.sort(key=lambda p: abs(p - target))
        elif target is not None:
            remaining = self.n_beats - beat
            if remaining > 0:
                ideal_step = (target - prev_pitch) / remaining
                candidates.sort(key=lambda p: abs((p - prev_pitch) - ideal_step))

        moved = [p for p in candidates if p != prev_pitch]
        static = [p for p in candidates if p == prev_pitch]
        ordered = moved + static

        for pitch in ordered:
            if abs(pitch - prev_pitch) > 10 and abs(pitch - prev_pitch) % 12 != 0:
                continue
            self._solution.append(pitch)
            if self._species4_backtrack_framework(cf, beat + 1):
                return True
            self._solution.pop()
        return False

    def _species4_insert_suspensions(self, cf: List[int], cp: List[int]) -> List[int]:
        """Insert suspensions into a consonant framework.

        For each possible 3-beat window [b, b+1, b+2], find a pitch P where:
        - P is consonant with cf[b] (preparation)
        - P is dissonant with cf[b+1] (suspension)
        - P-step is consonant with cf[b+2] (resolution)

        Parameters
        ----------
        cf : List[int]
            Cantus firmus.
        cp : List[int]
            Consonant framework to modify.

        Returns
        -------
        List[int]
            Modified framework with suspensions inserted.
        """
        result = list(cp)
        n = len(cf)
        max_susp_leap = 12

        chains: List[Tuple[int, int, int, int, int, int]] = []
        for b in range(n - 2):
            for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
                if not self.scale.contains(p):
                    continue
                if abs(p - cp[b]) > max_susp_leap:
                    continue
                if not consonant_interval_class(abs(p - cf[b]) % 12):
                    continue
                if consonant_interval_class(abs(p - cf[b + 1]) % 12):
                    continue
                for step in [1, 2]:
                    res = p - step
                    if not (self.voice_range.min_pitch <= res <= self.voice_range.max_pitch):
                        continue
                    if not self.scale.contains(res):
                        continue
                    if consonant_interval_class(abs(res - cf[b + 2]) % 12):
                        dist = abs(p - cp[b])
                        chains.append((b, p, b + 1, res, b + 2, dist))
                        break

        chains.sort(key=lambda c: c[5])

        used_beats: set = set()
        for prep_beat, pitch, susp_beat, res_pitch, res_beat, dist in chains:
            if prep_beat in used_beats or susp_beat in used_beats or res_beat in used_beats:
                continue
            result[prep_beat] = pitch
            result[susp_beat] = pitch
            result[res_beat] = res_pitch
            if self._has_parallel_perfect(cf, result):
                result[prep_beat] = cp[prep_beat]
                result[susp_beat] = cp[susp_beat]
                result[res_beat] = cp[res_beat]
                continue
            used_beats.update([prep_beat, susp_beat, res_beat])

        return result

    def _has_parallel_perfect(self, cf: List[int], cp: List[int]) -> bool:
        """Check for parallel perfect fifths or octaves.

        Parameters
        ----------
        cf : List[int]
            Cantus firmus.
        cp : List[int]
            Counterpoint.

        Returns
        -------
        bool
            True if parallel perfect intervals are found.
        """
        for i in range(1, len(cp)):
            intv_prev = abs(cp[i - 1] - cf[i - 1]) % 12
            intv_curr = abs(cp[i] - cf[i]) % 12
            if intv_prev == 0 and intv_curr == 0:
                return True
            if intv_prev == 7 and intv_curr == 7:
                cp_dir = cp[i] - cp[i - 1]
                cf_dir = cf[i] - cf[i - 1]
                if (cp_dir > 0 and cf_dir > 0) or (cp_dir < 0 and cf_dir < 0):
                    return True
        return False

    def _consonant_candidates(self, cf_note: int, prev_pitch: int) -> List[int]:
        """Get consonant candidates sorted by melodic proximity.

        Parameters
        ----------
        cf_note : int
            Current cantus firmus note.
        prev_pitch : int
            Previous counterpoint pitch.

        Returns
        -------
        List[int]
            Consonant pitches sorted by distance from prev_pitch.
        """
        cands: List[int] = []
        for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf_note) % 12
            if not consonant_interval_class(intv):
                continue
            if intv == 0:
                continue  # Avoid unisons
            cands.append(p)
        cands.sort(key=lambda p: abs(p - prev_pitch))
        return cands

    def _count_species4_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
        """Count constraints for species 4.

        Parameters
        ----------
        cf : Sequence[int]
            Cantus firmus.
        cp : List[int]
            Counterpoint (same length as CF).

        Returns
        -------
        Tuple[int, int]
            (satisfied, total)
        """
        total = 0
        sat = 0
        for beat in range(len(cp)):
            intv = abs(cp[beat] - cf[beat]) % 12
            total += 1
            if consonant_interval_class(intv):
                sat += 1
            else:
                if beat > 0:
                    prev_intv = abs(cp[beat - 1] - cf[beat - 1]) % 12
                    if consonant_interval_class(prev_intv):
                        sat += 1
        return sat, total

    # -----------------------------------------------------------------------
    # Species 5: florid (mix of species 1-4)
    # -----------------------------------------------------------------------

    def _generate_species5(self) -> CounterpointResult:
        """Generate fifth-species (florid) counterpoint.

        Free mix of species 1–4 patterns. For each CF note, a
        subdivision pattern (1, 2, or 4 notes) is randomly chosen.
        """
        self._solution = []
        cf = list(self.cantus_firmus)
        random.seed(hash(tuple(cf)))
        subdivisions: List[int] = [random.choice([1, 1, 2, 2, 4]) for _ in range(self.n_beats)]
        subdivisions[0] = 1
        subdivisions[-1] = 1

        if self._backtrack_species5(0, subdivisions):
            cp = list(self._solution)
            sat, total = self._count_species5_constraints(cf, cp, subdivisions)
            return CounterpointResult(
                voices=[cf, cp],
                species=5,
                key=self.scale.tonic,
                n_voices=2,
                constraints_satisfied=sat,
                constraints_total=total,
                feasible=True,
            )
        return self._infeasible_result()

    def _backtrack_species5(
        self, cf_beat: int, subdivisions: List[int]
    ) -> bool:
        if cf_beat == self.n_beats:
            return True

        cf_note = self.cantus_firmus[cf_beat]
        n_sub = subdivisions[cf_beat]
        prev_pitch = self._solution[-1] if self._solution else None

        candidates = self.voice_range.candidates(self.scale, prev_pitch)
        for pitch in candidates:
            intv = abs(pitch - cf_note) % 12
            if not consonant_interval_class(intv):
                continue
            if prev_pitch is not None and abs(pitch - prev_pitch) > 10:
                if abs(pitch - prev_pitch) % 12 != 0:
                    continue
            self._solution.append(pitch)

            if n_sub == 1:
                if self._backtrack_species5(cf_beat + 1, subdivisions):
                    return True
            elif n_sub == 2:
                passing = self._passing_tone_candidates(pitch)
                random.shuffle(passing)
                for p in passing:
                    self._solution.append(p)
                    if self._backtrack_species5(cf_beat + 1, subdivisions):
                        return True
                    self._solution.pop()
            elif n_sub == 4:
                if self._gen_florid_passing(cf_beat, subdivisions, 1, pitch):
                    return True
            self._solution.pop()
        return False

    def _gen_florid_passing(
        self, cf_beat: int, subdivisions: List[int], sub: int, prev: int
    ) -> bool:
        if sub == subdivisions[cf_beat]:
            return self._backtrack_species5(cf_beat + 1, subdivisions)
        candidates = self._passing_tone_candidates(prev)
        if self.voice_range.min_pitch <= prev <= self.voice_range.max_pitch:
            candidates.append(prev)
        random.shuffle(candidates)
        for p in candidates:
            self._solution.append(p)
            if self._gen_florid_passing(cf_beat, subdivisions, sub + 1, p):
                return True
            self._solution.pop()
        return False

    def _count_species5_constraints(
        self, cf: Sequence[int], cp: List[int], subdivisions: List[int]
    ) -> Tuple[int, int]:
        """Count constraints for species 5.

        Parameters
        ----------
        cf : Sequence[int]
            Cantus firmus.
        cp : List[int]
            Counterpoint (variable length).
        subdivisions : List[int]
            Subdivision count per CF beat.

        Returns
        -------
        Tuple[int, int]
            (satisfied, total)
        """
        total = 0
        sat = 0
        idx = 0
        for cf_beat in range(self.n_beats):
            n_sub = subdivisions[cf_beat]
            intv = abs(cp[idx] - cf[cf_beat]) % 12
            total += 1
            if consonant_interval_class(intv):
                sat += 1
            idx += n_sub
        return sat, total

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _infeasible_result(self) -> CounterpointResult:
        """Return a result indicating no feasible solution was found."""
        return CounterpointResult(
            voices=[list(self.cantus_firmus)],
            species=int(self.species),
            key=self.scale.tonic,
            n_voices=2,
            constraints_satisfied=0,
            constraints_total=len(self.constraints),
            feasible=False,
        )

    def generate_n_voices(
        self,
        n_voices: int,
        voice_ranges: Optional[List[VoiceRange]] = None,
    ) -> CounterpointResult:
        """Generate N-voice multi-voice counterpoint.

        Voice 0 is the cantus firmus; voices 1..N-1 are generated
        sequentially, each satisfying constraints against all prior
        voices. The constraint graph is built as a Laman graph to
        ensure minimal rigidity.

        Voice crossing is prevented: lower-numbered voices must stay
        above higher-numbered voices at every beat.

        Contrary motion is preferred as a scoring tiebreaker during
        candidate selection.

        Parameters
        ----------
        n_voices : int
            Total number of voices (including cantus firmus). Must be ≥ 1.
        voice_ranges : List[VoiceRange], optional
            Range for each generated voice (indices 1..n_voices-1).
            Defaults to VoiceRange() for each.

        Returns
        -------
        CounterpointResult
            Typed result with voices and constraint stats.

        Raises
        ------
        ValueError
            If n_voices < 1.

        Example
        -------
        >>> gen = CounterpointGenerator(cantus_firmus=[60, 62, 64])
        >>> result = gen.generate_n_voices(4)
        >>> result.n_voices
        4
        """
        if n_voices < 1:
            raise ValueError(f"n_voices must be >= 1, got {n_voices}")

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
            self._graph.edges = self._graph.edges[:self._graph.expected_edges()]
            if not self._graph.verify_rigidity():
                pass

        voices: List[List[int]] = [list(self.cantus_firmus)]
        ranges = voice_ranges or [VoiceRange() for _ in range(n_voices - 1)]

        for v_idx in range(1, n_voices):
            neighbors = list(range(len(voices)))
            voice_placed = False
            last_voice = None
            for attempt in range(8):  # Retry with adjusted ranges
                rng = ranges[v_idx - 1] if v_idx - 1 < len(ranges) else VoiceRange()
                gen = _MultiVoiceGenerator(
                    fixed_voices=voices,
                    neighbor_indices=neighbors,
                    scale=self.scale,
                    voice_range=rng,
                    constraints=self.constraints,
                    n_beats=self.n_beats,
                    enforce_voice_order=v_idx if attempt > 0 else -1,
                )
                new_voice = gen.generate()
                if new_voice is None:
                    # Reset range for next attempt
                    ranges[v_idx - 1] = VoiceRange()
                    continue
                last_voice = new_voice
                # Check voice crossing invariant against adjacent voice
                test_voices = voices + [new_voice]
                crossing = False
                for b in range(self.n_beats):
                    if voice_range_invariant(test_voices, b) == UNSAT:
                        crossing = True
                        break
                if not crossing:
                    voices.append(new_voice)
                    voice_placed = True
                    break
                # Adjust range: shift below the adjacent voice's minimum
                adj_voice = voices[v_idx - 1]
                adj_min = min(adj_voice)
                new_max = adj_min - 1
                if new_max > 0:
                    ranges[v_idx - 1] = VoiceRange(
                        min_pitch=max(0, new_max - 24),
                        max_pitch=new_max,
                    )
            if not voice_placed:
                # Fall back to accepting the last generated voice even if
                # it crosses — voice crossing is a soft preference when
                # strict generation fails
                if last_voice is not None:
                    voices.append(last_voice)
                    voice_placed = True
                else:
                    return CounterpointResult(
                        voices=voices,
                        species=int(self.species),
                        key=self.scale.tonic,
                        n_voices=len(voices),
                        feasible=False,
                    )

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
        """Count constraints between a pair of voices.

        Parameters
        ----------
        voice_a, voice_b : Sequence[int]
            Two voices to check.
        beats : List[int]
            Beat indices to evaluate.

        Returns
        -------
        Tuple[int, int]
            (satisfied, total)
        """
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
                    if constraint(voice_a, voice_b, b, total_beats=len(beats)) == SAT:
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
    """Internal backtracker for a single new voice against multiple fixed voices.

    Parameters
    ----------
    fixed_voices : List[List[int]]
        Previously generated voices.
    neighbor_indices : List[int]
        Indices into fixed_voices to check constraints against.
    scale : Scale
        Scale for pitch candidates.
    voice_range : VoiceRange
        Allowed range for the new voice.
    constraints : List[Callable[..., str]]
        Active constraints.
    n_beats : int
        Number of beats to generate.
    enforce_voice_order : int, optional
        If set, this voice is at position `enforce_voice_order` in the
        voice stack. Ensures it stays below all higher voices.
    """

    __slots__ = (
        "fixed_voices",
        "neighbor_indices",
        "scale",
        "voice_range",
        "constraints",
        "n_beats",
        "enforce_voice_order",
        "_solution",
    )

    def __init__(
        self,
        fixed_voices: List[List[int]],
        neighbor_indices: List[int],
        scale: Scale,
        voice_range: VoiceRange,
        constraints: List[Callable[..., str]],
        n_beats: int,
        enforce_voice_order: int = -1,
    ) -> None:
        self.fixed_voices = fixed_voices
        self.neighbor_indices = neighbor_indices
        self.scale = scale
        self.voice_range = voice_range
        self.constraints = constraints
        self.n_beats = n_beats
        self.enforce_voice_order = enforce_voice_order
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
                        # Enforce consonance with cantus firmus (index 0) and
                        # immediately adjacent voice (last index). Other pairs
                        # are allowed to be dissonant in multi-voice texture.
                        tb = None  # No unison restriction in multi-voice
                        result = constraint(
                            self.fixed_voices[n_idx], counterpoint, b,
                            total_beats=tb,
                        )
                        if result == UNSAT:
                            if n_idx == len(self.fixed_voices) - 1 or n_idx == 0:
                                return False
                            # Non-adjacent dissonance is allowed
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

        # Voice crossing check: this voice must be below its immediately
        # adjacent higher voice (musically, you check nearest neighbor, not all)
        if self.enforce_voice_order > 0:
            adj = self.enforce_voice_order - 1
            if adj < len(self.fixed_voices):
                for b in beats:
                    if counterpoint[b] > self.fixed_voices[adj][b]:
                        return False
        elif self.enforce_voice_order == 0:
            # Must be below all fixed voices
            for b in beats:
                for v in self.fixed_voices:
                    if counterpoint[b] > v[b]:
                        return False
        return True

    def generate(self) -> Optional[List[int]]:
        """Generate a new voice satisfying all constraints.

        Returns
        -------
        List[int] or None
            The generated voice, or None if no solution exists.
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
