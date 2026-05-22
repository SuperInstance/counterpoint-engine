"""
Convert counterpoint voices to tensor-midi format.

Maps:
- Voices → MIDI channels (0-15)
- Intervals → FluxVector directions (dodecet / vector48)
- Voice leading → Nod / Smile / Frown side channels

Output: TensorMIDIEvent objects compatible with flux-tensor-midi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from flux_tensor_midi.core.flux import FluxVector
from flux_tensor_midi.midi.events import MidiEvent
from constraint_theory_core.lattice import (
    A2Point,
    snap,
    encode_dodecet,
    decode_dodecet,
    vector48_encode,
    vector48_decode,
    DODECET_DIRECTIONS,
)


# ---------------------------------------------------------------------------
# TensorMIDIEvent — the 4-byte phase-state event from the theory
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TensorMIDIEvent:
    """A 4-byte Tensor-MIDI event encoding phase state.

    Attributes
    ----------
    cos_int8 : int
        Phase direction X, saturated INT8.
    sin_int8 : int
        Phase direction Y, saturated INT8.
    beat_k : int
        Beat counter 0-255 (wraps).
    state_byte : int
        Agent state as INT8 (encodes side-channel gestures).
    """

    cos_int8: int
    sin_int8: int
    beat_k: int
    state_byte: int

    def __post_init__(self):
        # Saturate to INT8 range
        object.__setattr__(
            self, "cos_int8", max(-128, min(127, self.cos_int8))
        )
        object.__setattr__(
            self, "sin_int8", max(-128, min(127, self.sin_int8))
        )
        object.__setattr__(
            self, "beat_k", max(0, min(255, self.beat_k))
        )
        object.__setattr__(
            self, "state_byte", max(0, min(255, self.state_byte))
        )

    def to_bytes(self) -> bytes:
        """Return as 4 raw bytes."""
        return bytes([
            self.cos_int8 & 0xFF,
            self.sin_int8 & 0xFF,
            self.beat_k,
            self.state_byte,
        ])

    @classmethod
    def from_pitch_interval(
        cls,
        pitch: int,
        interval: int,
        beat: int,
        side_state: int = 0,
    ) -> "TensorMIDIEvent":
        """Create a TensorMIDIEvent from pitch and interval.

        The interval determines the phase direction via the A₂ lattice.
        The pitch modulates the salience (encoded in state_byte nibble).
        """
        # Map interval (0-11) to one of 12 dodecet directions
        dodecet_idx = interval % 12
        dir_a, dir_b = DODECET_DIRECTIONS[dodecet_idx]
        # Scale to INT8 range (roughly)
        cos_i8 = int(dir_a * 60)
        sin_i8 = int(dir_b * 60)
        # State byte: high nibble = side_state, low nibble = pitch class
        state = (side_state & 0x0F) << 4 | (pitch % 12)
        return cls(cos_i8, sin_i8, beat % 256, state)

    def __repr__(self) -> str:
        return (
            f"TensorMIDIEvent(cos={self.cos_int8}, sin={self.sin_int8}, "
            f"beat={self.beat_k}, state=0x{self.state_byte:02x})"
        )


# ---------------------------------------------------------------------------
# Voice → Tensor mapping
# ---------------------------------------------------------------------------

def voices_to_tensor_events(
    voices: Sequence[Sequence[int]],
    beat_duration_ms: float = 500.0,
    velocity: int = 100,
) -> Tuple[List[TensorMIDIEvent], List[MidiEvent]]:
    """Convert a set of counterpoint voices to tensor-midi events.

    Parameters
    ----------
    voices : Sequence[Sequence[int]]
        List of voices, each a sequence of MIDI note numbers.
    beat_duration_ms : float, default 500.0
        Duration of one beat in milliseconds.
    velocity : int, default 100
        MIDI note velocity.

    Returns
    -------
    (tensor_events, midi_events)
        Parallel lists of TensorMIDIEvent and MidiEvent objects.
    """
    tensor_events: List[TensorMIDIEvent] = []
    midi_events: List[MidiEvent] = []

    n_beats = len(voices[0]) if voices else 0

    for beat in range(n_beats):
        start_ms = beat * beat_duration_ms
        # Compute intervals from bass (voice 0) to each upper voice
        bass_pitch = voices[0][beat]
        for v_idx, voice in enumerate(voices):
            pitch = voice[beat]
            channel = v_idx  # map voice index directly to MIDI channel

            # MIDI event
            midi_events.append(
                MidiEvent(
                    note=pitch,
                    velocity=velocity,
                    start_ms=start_ms,
                    duration_ms=beat_duration_ms,
                    channel=channel,
                )
            )

            # Interval from bass (or unison for bass itself)
            if v_idx == 0:
                interval = 0
            else:
                interval = abs(pitch - bass_pitch) % 12

            # Side-state encoding: encode voice-leading quality
            side_state = _voice_leading_side_state(voices, v_idx, beat)

            tensor_events.append(
                TensorMIDIEvent.from_pitch_interval(
                    pitch=pitch,
                    interval=interval,
                    beat=beat,
                    side_state=side_state,
                )
            )

    return tensor_events, midi_events


def _voice_leading_side_state(
    voices: Sequence[Sequence[int]],
    voice_idx: int,
    beat: int,
) -> int:
    """Encode voice-leading quality as a 4-bit side state.

    0 = Nod (stable, stepwise or small leap)
    1 = Smile (good consonance, strong harmonic position)
    2 = Frown (dissonance or large leap)
    3 = Resolve (leading-tone resolution)
    """
    voice = voices[voice_idx]
    if beat == 0:
        return 1  # Smile on opening

    prev = voice[beat - 1]
    curr = voice[beat]
    leap = abs(curr - prev)
    simple_leap = leap % 12

    # Check for leading-tone resolution
    if (prev % 12) == 11 and (curr % 12) == 0:
        return 3  # Resolve

    if simple_leap <= 2:
        return 0  # Nod — stepwise is stable
    if simple_leap in (3, 4, 7, 8, 9):
        return 1  # Smile — consonant leap
    return 2  # Frown — dissonant or large leap


# ---------------------------------------------------------------------------
# Explicit side-channel helpers
# ---------------------------------------------------------------------------

def voice_leading_to_sidechannels(
    voices: Sequence[Sequence[int]],
    beat: int,
) -> Dict[Tuple[int, int], str]:
    """Map voice-leading motion between all pairs to side-channel gestures.

    Returns a dict mapping (voice_i, voice_j) to "Nod", "Smile", or "Frown".

    Parameters
    ----------
    voices : Sequence[Sequence[int]]
        Voices as MIDI note sequences.
    beat : int
        Beat index to evaluate (checks motion from beat-1 to beat).

    Returns
    -------
    Dict[Tuple[int, int], str]
    """
    result: Dict[Tuple[int, int], str] = {}
    if beat == 0:
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                result[(i, j)] = "Nod"
        return result

    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            vi_prev = voices[i][beat - 1]
            vi_curr = voices[i][beat]
            vj_prev = voices[j][beat - 1]
            vj_curr = voices[j][beat]

            motion_i = vi_curr - vi_prev
            motion_j = vj_curr - vj_prev

            # Contrary motion = Smile (good independence)
            # Similar stepwise = Nod (acceptable)
            # Similar large leap or parallel perfect = Frown
            if motion_i == 0 or motion_j == 0:
                result[(i, j)] = "Nod"
            elif (motion_i > 0 and motion_j < 0) or (motion_i < 0 and motion_j > 0):
                result[(i, j)] = "Smile"
            else:
                # Similar motion — check interval
                interval = abs(vi_curr - vj_curr) % 12
                if interval in (0, 7):
                    result[(i, j)] = "Frown"
                elif abs(motion_i) <= 2 and abs(motion_j) <= 2:
                    result[(i, j)] = "Nod"
                else:
                    result[(i, j)] = "Frown"

    return result


def interval_to_flux_vector(interval_semitones: int) -> FluxVector:
    """Map a musical interval to a FluxVector direction.

    Uses the A₂ lattice dodecet: 12 directions mapped to interval classes.
    The interval class determines which channel is active.
    """
    interval_class = interval_semitones % 12
    values = [0.0] * 9
    # Map interval class 0-11 into channels 0-8 (cycle)
    channel = interval_class % 9
    # Strength decreases with dissonance
    strength = 1.0 if interval_class in (0, 3, 4, 7, 8, 9) else 0.5
    values[channel] = strength
    return FluxVector(values)


def voice_intervals_to_flux_vectors(
    voices: Sequence[Sequence[int]],
    beat: int,
) -> List[FluxVector]:
    """Return FluxVector for each voice at a given beat.

    Each vector encodes the interval from the bass voice as a direction.
    """
    if not voices:
        return []
    bass = voices[0][beat]
    vectors: List[FluxVector] = []
    for voice in voices:
        interval = abs(voice[beat] - bass)
        vectors.append(interval_to_flux_vector(interval))
    return vectors
