"""Canon generation — imitative counterpoint at various intervals and offsets.

A canon takes a *leader* melody and produces one or more *follower* voices
that repeat the leader at a fixed pitch interval and time offset (delay).
The result must still satisfy basic contrapuntal constraints (no parallel
perfect intervals, consonant strong beats, etc.).

This module provides:

- :class:`CanonVoice` — dataclass describing a single follower voice.
- :class:`CanonGenerator` — builds multi-voice canons from a leader melody.
- :func:`make_follower` — convenience function for a single follower.

No external dependencies beyond the standard library and ``counterpoint_engine``
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from counterpoint_engine.generator import (
    CounterpointGenerator,
    CounterpointResult,
    Scale,
    Species,
    VoiceRange,
)
from counterpoint_engine.rules import (
    SAT,
    consonant_interval_class,
    no_parallel_fifths,
    no_parallel_octaves,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CanonVoice:
    """Configuration for a single follower voice in a canon.

    Attributes
    ----------
    interval : int
        Transposition interval in semitones (positive = higher).
        Common values: 0 (unison), 4 (at the fourth above), 5 (at the
        fifth above), -5 (at the fourth below), -7 (at the fifth below).
    offset : int
        Time offset in beats — how many beats the follower lags behind
        the leader. Must be >= 1 for a true canon.
    voice_range : VoiceRange
        Allowed pitch range for this follower voice.
    label : str
        Human-readable label (e.g. ``"comes"``).
    """

    interval: int = 4
    offset: int = 2
    voice_range: VoiceRange = field(default_factory=VoiceRange)
    label: str = "follower"

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"offset must be >= 0, got {self.offset}")


@dataclass(frozen=True, slots=True)
class CanonResult:
    """Output of canon generation.

    Attributes
    ----------
    leader : List[int]
        Original leader melody (may be padded/rested at the start).
    voices : List[List[int]]
        All voices including the leader at index 0.
    voice_configs : List[CanonVoice]
        Configuration used for each follower voice.
    total_beats : int
        Length of each voice (padded with rests to accommodate offsets).
    feasible : bool
        Whether all constraints were satisfied.
    constraint_score : float
        Fraction of constraint checks that passed (0.0–1.0).
    """

    leader: List[int]
    voices: List[List[int]]
    voice_configs: List[CanonVoice]
    total_beats: int
    feasible: bool
    constraint_score: float = 1.0


# Sentinel for a rest (silence)
REST: int = -1


# ---------------------------------------------------------------------------
# Canon generator
# ---------------------------------------------------------------------------

@dataclass
class CanonGenerator:
    """Generate a multi-voice canon from a leader melody.

    The generator transposes the leader melody by the configured interval
    and shifts it by the configured offset to create each follower voice.
    It then validates the result against basic counterpoint constraints
    (no parallel fifths/octaves, consonance at downbeats).

    Parameters
    ----------
    leader : Sequence[int]
        The leader melody (MIDI note numbers). Must not be empty.
    scale : Scale
        Diatonic scale for pitch validation.
    followers : List[CanonVoice]
        Configuration for each follower voice.
    validate : bool
        If True, run constraint checks on the result and report
        ``feasible`` / ``constraint_score``.

    Example
    -------
    >>> leader = [60, 62, 64, 65, 67, 65, 64, 62]
    >>> cg = CanonGenerator(
    ...     leader=leader,
    ...     followers=[CanonVoice(interval=4, offset=2)],
    ... )
    >>> result = cg.generate()
    >>> result.feasible
    True
    >>> len(result.voices)
    2
    """

    leader: Sequence[int]
    scale: Scale = field(default_factory=Scale)
    followers: List[CanonVoice] = field(default_factory=lambda: [CanonVoice()])
    validate: bool = True

    def __post_init__(self) -> None:
        if not self.leader:
            raise ValueError("leader melody must not be empty")
        for i, note in enumerate(self.leader):
            if not isinstance(note, int):
                raise TypeError(
                    f"leader[{i}] must be int, got {type(note).__name__}"
                )

    def generate(self) -> CanonResult:
        """Generate the canon and return a :class:`CanonResult`.

        Returns
        -------
        CanonResult
            The generated canon with all voices and validation results.
        """
        max_offset = max((f.offset for f in self.followers), default=0)
        total_beats = len(self.leader) + max_offset

        # Build leader voice (padded with RESTs at the end)
        leader_voice: List[int] = list(self.leader) + [REST] * max_offset

        # Build each follower voice
        follower_voices: List[List[int]] = []
        for cfg in self.followers:
            follower = self._build_follower(cfg, total_beats)
            follower_voices.append(follower)

        all_voices = [leader_voice] + follower_voices

        # Validate
        feasible = True
        score = 1.0
        if self.validate:
            feasible, score = self._validate(all_voices)

        return CanonResult(
            leader=leader_voice,
            voices=all_voices,
            voice_configs=list(self.followers),
            total_beats=total_beats,
            feasible=feasible,
            constraint_score=score,
        )

    def _build_follower(self, cfg: CanonVoice, total_beats: int) -> List[int]:
        """Construct a single follower voice.

        Parameters
        ----------
        cfg : CanonVoice
            Configuration for this follower.
        total_beats : int
            Total length of the output voice.

        Returns
        -------
        List[int]
            The follower voice (REST-padded).
        """
        voice: List[int] = [REST] * cfg.offset
        for note in self.leader:
            if note == REST:
                voice.append(REST)
            else:
                transposed = note + cfg.interval
                voice.append(transposed)
        # Pad to total_beats
        while len(voice) < total_beats:
            voice.append(REST)
        return voice[:total_beats]

    def _validate(self, voices: List[List[int]]) -> Tuple[bool, float]:
        """Validate the canon against counterpoint constraints.

        Checks all voice pairs for:
        - Parallel perfect fifths and octaves
        - Consonance at overlapping (non-REST) beats

        Parameters
        ----------
        voices : List[List[int]]
            All voices (leader + followers).

        Returns
        -------
        Tuple[bool, float]
            (feasible, score) where feasible is True if no hard violations,
            and score is the fraction of checks that passed.
        """
        total_checks = 0
        passed_checks = 0

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                va = voices[i]
                vb = voices[j]
                # Find overlapping non-REST beats
                beats: List[int] = [
                    b for b in range(len(va))
                    if va[b] != REST and vb[b] != REST
                ]
                if len(beats) < 2:
                    continue

                # Check parallel fifths
                total_checks += 1
                if no_parallel_fifths(va, vb, beats) == SAT:
                    passed_checks += 1

                # Check parallel octaves
                total_checks += 1
                if no_parallel_octaves(va, vb, beats) == SAT:
                    passed_checks += 1

                # Check consonance at first and last overlapping beat
                for b in (beats[0], beats[-1]):
                    intv = abs(va[b] - vb[b]) % 12
                    total_checks += 1
                    if consonant_interval_class(intv):
                        passed_checks += 1

        score = passed_checks / total_checks if total_checks > 0 else 1.0
        feasible = score >= 1.0
        return feasible, score

    def generate_with_counterpoint(self) -> CounterpointResult:
        """Generate a canon using the full :class:`CounterpointGenerator`.

        This uses the leader as a cantus firmus and generates counterpoint
        against it, then overlays the result canon-style.

        Returns
        -------
        CounterpointResult
            Result from the underlying generator.

        Note
        ----
        This method is useful when you want the leader to be a strict
        cantus firmus and generate a species-counterpoint counterpoint
        that follows canon rules.
        """
        gen = CounterpointGenerator(
            cantus_firmus=list(self.leader),
            species=Species.FIRST,
            scale=self.scale,
        )
        return gen.generate()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def make_follower(
    leader: Sequence[int],
    interval: int = 4,
    offset: int = 2,
    voice_range: Optional[VoiceRange] = None,
) -> CanonResult:
    """Generate a single-follower canon.

    Parameters
    ----------
    leader : Sequence[int]
        The leader melody.
    interval : int
        Transposition interval in semitones.
    offset : int
        Time offset in beats.
    voice_range : VoiceRange, optional
        Allowed range for the follower.

    Returns
    -------
    CanonResult
        The generated canon.

    Example
    -------
    >>> result = make_follower([60, 62, 64, 65], interval=7, offset=1)
    >>> len(result.voices)
    2
    """
    vr = voice_range or VoiceRange()
    cfg = CanonVoice(interval=interval, offset=offset, voice_range=vr)
    cg = CanonGenerator(leader=leader, followers=[cfg])
    return cg.generate()


def round_canon(
    leader: Sequence[int],
    n_voices: int = 4,
    offset: int = 2,
    scale: Optional[Scale] = None,
) -> CanonResult:
    """Generate a round (canon at the unison) with *n_voices* entries.

    Each voice enters *offset* beats after the previous one, at the
    same pitch level (interval = 0).

    Parameters
    ----------
    leader : Sequence[int]
        The leader melody.
    n_voices : int
        Total number of voices (including leader).
    offset : int
        Entry interval in beats between consecutive voices.
    scale : Scale, optional
        Scale for validation.

    Returns
    -------
    CanonResult
        The generated round.

    Example
    -------
    >>> result = round_canon([60, 62, 64, 65], n_voices=3, offset=2)
    >>> len(result.voices)
    3
    """
    followers = [
        CanonVoice(interval=0, offset=offset * (i + 1), label=f"voice_{i+1}")
        for i in range(n_voices - 1)
    ]
    cg = CanonGenerator(
        leader=leader,
        followers=followers,
        scale=scale or Scale(),
    )
    return cg.generate()
