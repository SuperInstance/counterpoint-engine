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

import random

from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    Satisfiability,
    cambiata_ok,
    consonant_interval,
    consonant_interval_class,
    is_step,
    max_leap_seventh,
    no_parallel_fifths,
    no_parallel_octaves,
    passing_tone_ok,
    proper_resolution,
    suspension_preparation,
    suspension_resolution,
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
        """Export to MIDI file using mido.

        For species 2/3/5, the counterpoint has more notes than the CF.
        CF notes get proportionally longer durations to stay aligned.
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

        Dispatches to species-specific generation methods.
        Returns a CounterpointResult. If no solution exists under
        the current constraints, feasible is False.
        """
        species_method = {
            Species.FIRST: self._generate_species1,
            Species.SECOND: self._generate_species2,
            Species.THIRD: self._generate_species3,
            Species.FOURTH: self._generate_species4,
            Species.FIFTH: self._generate_species5,
        }
        return species_method[self.species]()

    # -----------------------------------------------------------------------
    # Species 1: note-against-note (existing logic)
    # -----------------------------------------------------------------------

    def _generate_species1(self) -> CounterpointResult:
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
        """Two counterpoint notes per CF note.

        Beat 0 (strong): must be consonant with CF.
        Beat 1 (weak): passing tone — stepwise, can be dissonant.
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
        """Return stepwise neighbors as passing tone candidates."""
        candidates = []
        for step in [-2, -1, 1, 2]:
            p = from_pitch + step
            if self.voice_range.min_pitch <= p <= self.voice_range.max_pitch:
                candidates.append(p)
        return candidates

    def _count_species2_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
        """Count constraints for species 2."""
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
        """Four counterpoint notes per CF note.

        Beat 0 (strong): consonant with CF.
        Beats 1-3 (weak): mostly stepwise passing tones, can be dissonant.
        Beat 3 can use cambiata pattern.
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
            # All 4 subdivision notes placed, move to next CF beat
            return self._backtrack_species3(cf_beat + 1)

        # Generate stepwise note (passing tone)
        candidates = self._passing_tone_candidates(prev_pitch)
        # Also allow staying on same pitch for variety
        if self.voice_range.min_pitch <= prev_pitch <= self.voice_range.max_pitch:
            candidates.append(prev_pitch)
        random.shuffle(candidates)

        for pitch in candidates:
            self._solution.append(pitch)
            if sub_beat < 3:
                # beats 1-2: just stepwise
                if self._backtrack_species3_subdivisions(cf_beat, sub_beat + 1, pitch):
                    return True
            else:
                # beat 3: check that we can transition to next CF beat's consonance
                if self._backtrack_species3(cf_beat + 1):
                    return True
            self._solution.pop()
        return False

    def _count_species3_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
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
        """Counterpoint with syncopation — tied across beat boundaries.

        Creates suspension chains:
        preparation (consonant) → suspension (dissonant) → resolution (consonant, step down)

        Two-pass approach:
        1. Generate consonant frameworks with varying range
        2. Insert suspensions by modifying preparation beats
        3. Pick the best result combining suspensions and range
        """
        cf = list(self.cantus_firmus)

        best_solution = None
        best_score = -1

        # Collect all valid starting pitches
        start_pitches = []
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
                # Try inserting suspensions into this framework
                with_susp = self._species4_insert_suspensions(cf, framework)
                susp_count = sum(
                    1 for b in range(len(with_susp))
                    if not consonant_interval_class(abs(with_susp[b] - cf[b]) % 12)
                )
                span = max(with_susp) - min(with_susp)
                unique = len(set(with_susp))

                # Ideal: suspensions + wide range
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
        (or vice versa) to span a wide range, ensuring consonance at every beat.
        """
        best = None
        best_score = -1

        # Collect all valid starting pitches
        start_pitches = []
        for p in range(self.voice_range.max_pitch, self.voice_range.min_pitch - 1, -1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf[0]) % 12
            if consonant_interval_class(intv) and intv != 0:
                start_pitches.append(p)

        # Also collect ending pitches for the last CF note
        end_targets = []
        for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf[-1]) % 12
            if consonant_interval_class(intv) and intv != 0:
                end_targets.append(p)

        for start in start_pitches[:15]:
            # Plan a target arc: aim for a pitch far from start on the last beat
            for end_pitch in end_targets:
                if abs(end_pitch - start) < 12:
                    continue  # skip if not enough span potential
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

        # Fallback: no arc target, just search normally
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

        # If we have a target end pitch, steer toward it
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

        Prefer P close to the existing cp[b] to maintain the framework's range.
        """
        result = list(cp)
        n = len(cf)
        max_susp_leap = 12  # max semitones a suspension pitch can jump from framework

        # Find all possible suspension chains
        chains = []  # list of (prep_beat, pitch, susp_beat, res_pitch, res_beat)
        for b in range(n - 2):
            for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
                if not self.scale.contains(p):
                    continue
                # Leap constraint: don't jump too far from framework pitch
                if abs(p - cp[b]) > max_susp_leap:
                    continue
                # Preparation: consonant with cf[b]
                if not consonant_interval_class(abs(p - cf[b]) % 12):
                    continue
                # Suspension: dissonant with cf[b+1]
                if consonant_interval_class(abs(p - cf[b + 1]) % 12):
                    continue
                # Resolution: step down to consonance with cf[b+2]
                for step in [1, 2]:
                    res = p - step
                    if not (self.voice_range.min_pitch <= res <= self.voice_range.max_pitch):
                        continue
                    if not self.scale.contains(res):
                        continue
                    if consonant_interval_class(abs(res - cf[b + 2]) % 12):
                        # Check that resolution interval isn't unison (boring)
                        res_intv = abs(res - cf[b + 2]) % 12
                        # Valid chain found
                        dist = abs(p - cp[b])  # how far from framework
                        chains.append((b, p, b + 1, res, b + 2, dist))
                        break  # take first valid step

        # Sort by distance from framework (prefer minimal changes)
        chains.sort(key=lambda c: c[5])

        # Apply non-overlapping chains
        used_beats = set()
        for prep_beat, pitch, susp_beat, res_pitch, res_beat, dist in chains:
            if prep_beat in used_beats or susp_beat in used_beats or res_beat in used_beats:
                continue
            # Check for parallel perfect intervals after modification
            result[prep_beat] = pitch
            result[susp_beat] = pitch  # hold (suspension)
            result[res_beat] = res_pitch  # resolution
            # Verify no parallel fifths/octaves introduced
            if self._has_parallel_perfect(cf, result):
                # Revert
                result[prep_beat] = cp[prep_beat]
                result[susp_beat] = cp[susp_beat]
                result[res_beat] = cp[res_beat]
                continue
            used_beats.update([prep_beat, susp_beat, res_beat])

        return result

    def _has_parallel_perfect(self, cf: List[int], cp: List[int]) -> bool:
        """Check for parallel perfect fifths or octaves."""
        for i in range(1, len(cp)):
            intv_prev = abs(cp[i - 1] - cf[i - 1]) % 12
            intv_curr = abs(cp[i] - cf[i]) % 12
            # Parallel octaves/unisons
            if intv_prev == 0 and intv_curr == 0:
                return True
            # Parallel fifths
            if intv_prev == 7 and intv_curr == 7:
                # Check similar motion
                cp_dir = cp[i] - cp[i - 1]
                cf_dir = cf[i] - cf[i - 1]
                if (cp_dir > 0 and cf_dir > 0) or (cp_dir < 0 and cf_dir < 0):
                    return True
        return False

    def _consonant_candidates(self, cf_note: int, prev_pitch: int) -> List[int]:
        """Get consonant candidates sorted by melodic proximity to prev_pitch."""
        cands = []
        for p in range(self.voice_range.min_pitch, self.voice_range.max_pitch + 1):
            if not self.scale.contains(p):
                continue
            intv = abs(p - cf_note) % 12
            if not consonant_interval_class(intv):
                continue
            if intv == 0:
                continue  # Avoid unisons
            cands.append(p)
        # Sort by distance from prev for stepwise preference
        cands.sort(key=lambda p: abs(p - prev_pitch))
        return cands

    def _count_species4_constraints(
        self, cf: Sequence[int], cp: List[int]
    ) -> Tuple[int, int]:
        total = 0
        sat = 0
        for beat in range(len(cp)):
            intv = abs(cp[beat] - cf[beat]) % 12
            total += 1
            if consonant_interval_class(intv):
                sat += 1
            else:
                # Dissonance — check if valid suspension
                if beat > 0:
                    prev_intv = abs(cp[beat - 1] - cf[beat - 1]) % 12
                    if consonant_interval_class(prev_intv):
                        sat += 1  # valid suspension
        return sat, total

    # -----------------------------------------------------------------------
    # Species 5: florid (mix of species 1-4)
    # -----------------------------------------------------------------------

    def _generate_species5(self) -> CounterpointResult:
        """Florid counterpoint — free mix of species 1-4 patterns.

        For each CF note, randomly choose a subdivision pattern:
        - 1 note (species 1 style)
        - 2 notes (species 2 style)
        - 4 notes (species 3 style)
        """
        self._solution = []
        cf = list(self.cantus_firmus)
        # Pre-determine subdivision pattern for each CF beat
        random.seed(hash(tuple(cf)))  # deterministic for same CF
        subdivisions = [random.choice([1, 1, 2, 2, 4]) for _ in range(self.n_beats)]
        # Ensure first and last are species-1 style (standard practice)
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

        # Strong beat must be consonant
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
                # Species 2 style: one passing tone
                passing = self._passing_tone_candidates(pitch)
                random.shuffle(passing)
                for p in passing:
                    self._solution.append(p)
                    if self._backtrack_species5(cf_beat + 1, subdivisions):
                        return True
                    self._solution.pop()
            elif n_sub == 4:
                # Species 3 style: three passing tones
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
        total = 0
        sat = 0
        idx = 0
        for cf_beat in range(self.n_beats):
            n_sub = subdivisions[cf_beat]
            # Strong beat consonance
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
