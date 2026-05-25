#!/usr/bin/env python3
"""
tensor_output.py — Generate counterpoint, convert to TensorMIDIEvent objects.

Run:  python3 examples/tensor_output.py

Demonstrates:
  - Generating a 2-voice counterpoint
  - Converting voices to TensorMIDIEvent and MidiEvent streams
  - Inspecting the 4-byte phase-state encoding
  - Using voice_leading_to_sidechannels for gesture analysis
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counterpoint_engine import (
    CounterpointGenerator,
    Species,
    TensorMIDIEvent,
    voices_to_tensor_events,
    voice_leading_to_sidechannels,
)

DIVIDER = "=" * 60


def main():
    print(DIVIDER)
    print("Counterpoint → TensorMIDI Event Conversion")
    print(DIVIDER)

    # ── Generate counterpoint ───────────────────────────────────────────────
    cantus_firmus = [60, 62, 64, 65, 67, 69, 67, 65]  # C D E F G A G F

    print("\nCantus firmus:", cantus_firmus)
    print("Generating counterpoint...")

    gen = CounterpointGenerator(
        cantus_firmus=cantus_firmus,
        species=Species.FIRST,
    )
    counterpoint = gen.generate()

    if counterpoint is None:
        print("No counterpoint found — try a different cantus firmus.")
        return

    print(f"Counterpoint:  {counterpoint}")
    voices = [cantus_firmus, counterpoint]

    # ── Convert to TensorMIDIEvents ────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("TensorMIDIEvent stream")
    print(f"{'─' * 50}")
    print("\nEach event is 4 bytes: [cos_int8, sin_int8, beat_k, state_byte]")
    print("  cos/sin → phase direction from A₂ lattice dodecet")
    print("  beat_k  → beat counter (0–255)")
    print("  state   → voice-leading gesture (Nod/Smile/Frown/Resolve)\n")

    tensor_events, midi_events = voices_to_tensor_events(
        voices,
        beat_duration_ms=500.0,
        velocity=100,
    )

    for i, evt in enumerate(tensor_events):
        raw = evt.to_bytes()
        print(f"  [{i:2d}] {evt!r}  → raw bytes: {list(raw)}")

    # ── Show a few MidiEvents too ───────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("MidiEvent stream (first 8)")
    print(f"{'─' * 50}")
    for i, me in enumerate(midi_events[:8]):
        print(f"  [{i}] {me!r}")

    # ── Side-channel gesture analysis ──────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("Voice-leading side-channel gestures")
    print(f"{'─' * 50}")
    print("\nGestures: Nod=stable/stepwise, Smile=contrary motion,")
    print("          Frown=parallel/dissonant, Resolve=leading-tone→tonic\n")

    n_beats = len(voices[0])
    for beat in range(n_beats):
        channels = voice_leading_to_sidechannels(voices, beat)
        gestures = ", ".join(
            f"V{i}↔V{j}: {g}" for (i, j), g in channels.items()
        )
        print(f"  Beat {beat}: {gestures}")

    # ── Direct TensorMIDIEvent construction ─────────────────────────────────
    print(f"\n{'─' * 50}")
    print("Manual TensorMIDIEvent construction")
    print(f"{'─' * 50}")

    evt = TensorMIDIEvent.from_pitch_interval(
        pitch=67,       # G4
        interval=7,     # perfect fifth from bass
        beat=3,
        side_state=1,   # Smile
    )
    print("\n  from_pitch_interval(pitch=67, interval=7, beat=3):")
    print(f"    {evt!r}")
    print(f"    Raw bytes: {list(evt.to_bytes())}")

    print(f"\n✓ Generated {len(tensor_events)} tensor events from {len(voices)} voices × {n_beats} beats.")


if __name__ == "__main__":
    main()
