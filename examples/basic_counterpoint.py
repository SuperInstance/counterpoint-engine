#!/usr/bin/env python3
"""
basic_counterpoint.py — Generate first-species counterpoint above a cantus firmus.

Run:  python3 examples/basic_counterpoint.py

Demonstrates:
  - Building a cantus firmus in C major
  - Using CounterpointGenerator with Species.FIRST
  - Printing both voices side by side with interval labels
"""

import sys
import os

# Ensure the package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counterpoint_engine import (
    CounterpointGenerator,
    Species,
)

# ── Cantus firmus (MIDI note numbers) ──────────────────────────────────────
# A simple 8-note cantus firmus in C major: C D E F G A G F
CANTUS_FIRMUS = [60, 62, 64, 65, 67, 69, 67, 65]

# Note names for display (MIDI 60 = C4)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    """Convert a MIDI note number to a human-readable name like 'C4'."""
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def main():
    print("=" * 60)
    print("First-Species Counterpoint Generator")
    print("=" * 60)

    # ── Show the cantus firmus ──────────────────────────────────────────────
    cf_names = [midi_to_name(p) for p in CANTUS_FIRMUS]
    print(f"\nCantus firmus:  {' '.join(f'{n:>4}' for n in cf_names)}")
    print(f"MIDI notes:     {' '.join(f'{p:>4}' for p in CANTUS_FIRMUS)}")

    # ── Generate counterpoint ───────────────────────────────────────────────
    print("\nGenerating first-species counterpoint (note against note)...")
    print("Constraints: no parallel 5ths, no parallel octaves,")
    print("             consonant intervals, max leap 7th, proper resolution")

    gen = CounterpointGenerator(
        cantus_firmus=CANTUS_FIRMUS,
        species=Species.FIRST,
    )
    print(f"\n{gen!r}")

    counterpoint = gen.generate()

    if counterpoint is None:
        print("\nNo valid counterpoint found (constraints too strict for this CF).")
        return

    # ── Display results ─────────────────────────────────────────────────────
    cp_names = [midi_to_name(p) for p in counterpoint]

    print(f"\nCounterpoint:   {' '.join(f'{n:>4}' for n in cp_names)}")
    print(f"MIDI notes:     {' '.join(f'{p:>4}' for p in counterpoint)}")

    # Intervals between the two voices
    intervals = [abs(cp - cf) for cf, cp in zip(CANTUS_FIRMUS, counterpoint)]
    interval_names = []
    for iv in intervals:
        simple = iv % 12
        # Map semitone count to interval name
        names = {0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4",
                 6: "TT", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7"}
        compound = iv // 12
        base = names[simple]
        if compound > 0:
            interval_names.append(f"{base}+{compound * 8}")
        else:
            interval_names.append(f"{base:>4}")

    print(f"Intervals:      {' '.join(f'{n:>4}' for n in interval_names)}")

    # ── Show the staff-style view ───────────────────────────────────────────
    print("\n── Staff View ──")
    print("  Counterpoint (upper): ", "  ".join(f"{n:>3}" for n in cp_names))
    print("  ─────────────────────────────────────────")
    print("  Cantus firmus (lower): ", "  ".join(f"{n:>3}" for n in cf_names))

    print(f"\n✓ Generated {len(counterpoint)} notes of counterpoint above the cantus firmus.")


if __name__ == "__main__":
    main()
