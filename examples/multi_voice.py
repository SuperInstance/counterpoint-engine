#!/usr/bin/env python3
"""
multi_voice.py — Generate 3-voice and 4-voice counterpoint.

Demonstrates:
- Using generate_n_voices() for multi-part textures
- Each new voice is generated against all prior voices
- The constraint graph is built as a Laman graph for minimal rigidity

Run:  python3 examples/multi_voice.py
"""

from counterpoint_engine import (
    CounterpointGenerator,
    CounterpointResult,
    Species,
    VoiceRange,
    Scale,
    CounterpointGraph,
)

# Short cantus firmus in G major
CANTUS_FIRMUS = [67, 69, 71, 72, 74, 72, 71, 69, 67]

scale = Scale(tonic=7, mode="major")  # G major
gen = CounterpointGenerator(
    cantus_firmus=CANTUS_FIRMUS,
    species=Species.FIRST,
    scale=scale,
)

# --- 3-voice counterpoint ---
print("=" * 50)
print("3-VOICE COUNTERPOINT")
print("=" * 50)

ranges_3 = [
    VoiceRange(min_pitch=55, max_pitch=74),  # alto
    VoiceRange(min_pitch=48, max_pitch=67),  # tenor
]

result3 = gen.generate_n_voices(n_voices=3, voice_ranges=ranges_3)
print(f"Result: {result3}")
print(f"Feasible: {result3.feasible}")
print(f"Constraints: {result3.constraints_satisfied}/{result3.constraints_total}")

if result3.feasible:
    labels = ["Soprano (CF)", "Alto", "Tenor"]
    print()
    # Print header
    header = f"{'Beat':>4}"
    for label in labels:
        header += f"  {label:>12}"
    print(header)
    print("-" * len(header))
    for i in range(len(result3.voices[0])):
        row = f"{i:4d}"
        for v in result3.voices:
            row += f"  {v[i]:12d}"
        print(row)

# --- 4-voice counterpoint ---
print()
print("=" * 50)
print("4-VOICE COUNTERPOINT")
print("=" * 50)

ranges_4 = [
    VoiceRange(min_pitch=55, max_pitch=74),  # alto
    VoiceRange(min_pitch=48, max_pitch=67),  # tenor
    VoiceRange(min_pitch=36, max_pitch=55),  # bass
]

result4 = gen.generate_n_voices(n_voices=4, voice_ranges=ranges_4)
print(f"Result: {result4}")
print(f"Feasible: {result4.feasible}")
print(f"Constraints: {result4.constraints_satisfied}/{result4.constraints_total}")

if result4.feasible:
    labels4 = ["Soprano (CF)", "Alto", "Tenor", "Bass"]
    print()
    header = f"{'Beat':>4}"
    for label in labels4:
        header += f"  {label:>12}"
    print(header)
    print("-" * len(header))
    for i in range(len(result4.voices[0])):
        row = f"{i:4d}"
        for v in result4.voices:
            row += f"  {v[i]:12d}"
        print(row)

# --- Show the Laman constraint graph ---
print()
print("=" * 50)
print("LAMAN CONSTRAINT GRAPH (4 voices)")
print("=" * 50)
graph = CounterpointGraph(4)
print(f"Vertices: {graph.n_voices}")
print(f"Edges: {graph.edges}")
print(f"Edge count: {graph.edge_count()}")
print(f"Expected (2n-3): {graph.expected_edges()}")
print(f"Is minimally rigid: {graph.is_minimally_rigid()}")
print(f"Constraints per edge:")
for edge, names in graph.constraints.items():
    print(f"  {edge}: {names}")
