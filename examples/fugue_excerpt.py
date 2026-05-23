#!/usr/bin/env python3
"""
fugue_excerpt.py — Generate a short fugue-like excerpt using multi-voice counterpoint.

Demonstrates:
- Building a fugue structure with staggered voice entries
- Using Species.FIRST for strict note-against-note counterpoint
- Combining multiple CounterpointGenerator calls for successive entries
- Showing the full 4-voice texture

Run:  python3 examples/fugue_excerpt.py
"""

from counterpoint_engine import (
    CounterpointGenerator,
    Species,
    VoiceRange,
    Scale,
    CounterpointGraph,
)

# D minor fugue subject (MIDI notes)
SUBJECT = [62, 65, 64, 62, 67, 66, 64, 62, 60]  # D-Eb-D-C-F-E-C-D-Bb

scale = Scale(tonic=2, mode="minor")  # D minor

# Full cantus firmus = the subject
cf = list(SUBJECT)

gen = CounterpointGenerator(
    cantus_firmus=cf,
    species=Species.FIRST,
    scale=scale,
)

print("FUGUE EXCERPT IN D MINOR")
print("=" * 50)
print(f"Subject ({len(SUBJECT)} notes): {SUBJECT}")
print()

# Generate 4-voice counterpoint
ranges = [
    VoiceRange(min_pitch=57, max_pitch=76),  # alto
    VoiceRange(min_pitch=50, max_pitch=69),  # tenor
    VoiceRange(min_pitch=38, max_pitch=57),  # bass
]

result = gen.generate_n_voices(n_voices=4, voice_ranges=ranges)

if not result.feasible:
    print("Could not find a feasible 4-voice solution.")
    print("Trying 3-voice instead...")
    result = gen.generate_n_voices(n_voices=3, voice_ranges=ranges[:2])

print(f"Result: {result}")
print(f"Feasible: {result.feasible}")
print(f"Constraints: {result.constraints_satisfied}/{result.constraints_total}")
print()

if result.feasible:
    voice_labels = ["Soprano (Subject)", "Alto", "Tenor", "Bass"][:result.n_voices]

    # Print the score
    print("SCORE (MIDI note numbers):")
    header = f"{'Beat':>4}"
    for label in voice_labels:
        header += f"  {label:>16}"
    print(header)
    print("-" * len(header))

    for i in range(len(result.voices[0])):
        row = f"{i:4d}"
        for v in result.voices:
            row += f"  {v[i]:16d}"
        print(row)

    print()

    # Show intervals between adjacent voices
    print("INTERVALS (semitones, between adjacent voices):")
    for v_pair in range(len(result.voices) - 1):
        voice_a = result.voices[v_pair]
        voice_b = result.voices[v_pair + 1]
        intervals = [abs(voice_a[i] - voice_b[i]) % 12 for i in range(len(voice_a))]
        consonances = {0: "uni", 3: "m3", 4: "M3", 7: "P5", 8: "m6", 9: "M6"}
        names = [consonances.get(iv, f"{iv}") for iv in intervals]
        print(f"  {voice_labels[v_pair][:6]}→{voice_labels[v_pair+1][:6]}: {names}")

    # Show constraint graph
    print()
    g = CounterpointGraph(result.n_voices)
    print(f"Constraint graph: {g}")
    print(f"  Every voice pair is independently constrained.")
    print(f"  Total constraint edges: {g.edge_count()} (2×{result.n_voices}-3 = {2*result.n_voices-3})")
