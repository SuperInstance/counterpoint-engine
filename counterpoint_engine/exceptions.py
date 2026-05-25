"""Custom exceptions for counterpoint-specific errors.

These exceptions provide structured error information for constraint
violations (parallel fifths/octaves, voice crossing, range violations)
and input validation failures.
"""

from __future__ import annotations

from typing import Optional, Sequence


class CounterpointError(Exception):
    """Base exception for all counterpoint-engine errors."""

    def __init__(self, message: str, *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


# ---------------------------------------------------------------------------
# Constraint violations
# ---------------------------------------------------------------------------

class ConstraintViolationError(CounterpointError):
    """A contrapuntal constraint was violated during generation or checking."""

    def __init__(
        self,
        message: str,
        *,
        constraint: str = "",
        beat: int = -1,
        voices: Optional[Sequence[int]] = None,
        detail: str = "",
    ) -> None:
        self.constraint = constraint
        self.beat = beat
        self.voices = list(voices) if voices is not None else []
        super().__init__(message, detail=detail)


class ParallelFifthsError(ConstraintViolationError):
    """Parallel perfect fifths detected between two voices."""

    def __init__(
        self,
        voice_a: Sequence[int],
        voice_b: Sequence[int],
        beat_prev: int,
        beat_curr: int,
    ) -> None:
        self.voice_a = list(voice_a)
        self.voice_b = list(voice_b)
        self.beat_prev = beat_prev
        self.beat_curr = beat_curr
        super().__init__(
            f"Parallel fifths between beats {beat_prev} and {beat_curr}",
            constraint="no_parallel_fifths",
            beat=beat_curr,
            detail=(
                f"voice_a[{beat_prev}]={voice_a[beat_prev]}, "
                f"voice_a[{beat_curr}]={voice_a[beat_curr]}, "
                f"voice_b[{beat_prev}]={voice_b[beat_prev]}, "
                f"voice_b[{beat_curr}]={voice_b[beat_curr]}"
            ),
        )


class ParallelOctavesError(ConstraintViolationError):
    """Parallel perfect octaves detected between two voices."""

    def __init__(
        self,
        voice_a: Sequence[int],
        voice_b: Sequence[int],
        beat_prev: int,
        beat_curr: int,
    ) -> None:
        self.voice_a = list(voice_a)
        self.voice_b = list(voice_b)
        self.beat_prev = beat_prev
        self.beat_curr = beat_curr
        super().__init__(
            f"Parallel octaves between beats {beat_prev} and {beat_curr}",
            constraint="no_parallel_octaves",
            beat=beat_curr,
            detail=(
                f"voice_a[{beat_prev}]={voice_a[beat_prev]}, "
                f"voice_a[{beat_curr}]={voice_a[beat_curr]}, "
                f"voice_b[{beat_prev}]={voice_b[beat_prev]}, "
                f"voice_b[{beat_curr}]={voice_b[beat_curr]}"
            ),
        )


class VoiceCrossingError(ConstraintViolationError):
    """Voice crossing: a lower-numbered voice is above a higher-numbered one."""

    def __init__(
        self,
        voice_upper: int,
        voice_lower: int,
        beat: int,
        pitch_upper: int,
        pitch_lower: int,
    ) -> None:
        self.voice_upper_idx = voice_upper
        self.voice_lower_idx = voice_lower
        self.pitch_upper = pitch_upper
        self.pitch_lower = pitch_lower
        super().__init__(
            f"Voice crossing at beat {beat}: voice {voice_upper} "
            f"({pitch_upper}) below voice {voice_lower} ({pitch_lower})",
            constraint="voice_crossing",
            beat=beat,
        )


class RangeViolationError(ConstraintViolationError):
    """A pitch falls outside the allowed range for its voice."""

    def __init__(
        self,
        pitch: int,
        min_pitch: int,
        max_pitch: int,
        voice_index: int = -1,
        beat: int = -1,
    ) -> None:
        self.pitch = pitch
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.voice_index = voice_index
        super().__init__(
            f"Pitch {pitch} out of range [{min_pitch}, {max_pitch}]"
            + (f" for voice {voice_index}" if voice_index >= 0 else "")
            + (f" at beat {beat}" if beat >= 0 else ""),
            constraint="voice_range",
            beat=beat,
        )


class ResolutionError(ConstraintViolationError):
    """Leading tone failed to resolve to the tonic."""

    def __init__(
        self,
        voice: Sequence[int],
        beat: int,
        leading_tone: int,
        tonic: int,
    ) -> None:
        self.leading_tone = leading_tone
        self.tonic = tonic
        super().__init__(
            f"Leading tone at beat {beat - 1} ({voice[beat - 1]}) "
            f"did not resolve to tonic; got {voice[beat]} at beat {beat}",
            constraint="proper_resolution",
            beat=beat,
        )


class LeapViolationError(ConstraintViolationError):
    """Melodic leap exceeds the maximum allowed interval."""

    def __init__(
        self,
        prev_pitch: int,
        curr_pitch: int,
        max_leap: int,
        beat: int,
    ) -> None:
        self.prev_pitch = prev_pitch
        self.curr_pitch = curr_pitch
        self.actual_leap = abs(curr_pitch - prev_pitch)
        self.max_leap = max_leap
        super().__init__(
            f"Leap of {self.actual_leap} semitones at beat {beat} "
            f"exceeds maximum {max_leap}",
            constraint="max_leap_seventh",
            beat=beat,
        )


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------

class InvalidInputError(CounterpointError):
    """Raised when generator input fails validation."""

    def __init__(self, parameter: str, value: object, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        super().__init__(f"Invalid {parameter}={value!r}: {reason}")


class GenerationError(CounterpointError):
    """Raised when counterpoint generation fails entirely."""

    def __init__(
        self,
        message: str,
        *,
        species: int = 0,
        n_voices: int = 0,
        cantus_length: int = 0,
    ) -> None:
        self.species = species
        self.n_voices = n_voices
        self.cantus_length = cantus_length
        super().__init__(message)
