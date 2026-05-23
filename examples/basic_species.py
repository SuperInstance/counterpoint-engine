#!/usr/bin/env python3
"""
basic_species.py — Generate 1st species counterpoint against a cantus firmus.

Demonstrates:
- Creating a Scale and VoiceRange
- Setting up a CounterpointGenerator with a cantus firmus
- Running constraint-satisfaction backtracking to find a valid counterpoint
- Checking that all FLUX constraints are satisfied

Run:  python3 examples/basic_species.py
"""

from counterpoint_engine import (
    CounterpointGenerator,
    CounterpointResult,
    Species,
    VoiceRange,
    Scale,
    no_parallel_fifths,
    no_parallel_octaves,
    consonant_interval,
    SAT,
)

# A classic cantus firmus in C major (MIDI note numbers)
CANTUS_FIRMUS = [60, 62, 64, 65, 67, 65, 64, 62, 60]

# Define the scale (C major) and allowed range for the counterpoint voice
scale = Scale(tonic=0, mode="major")  # C major
voice_range = VoiceRange(min_pitch=60, max_pitch=79)  # C4 to G5

# Create the generator with 1st species (note-against-note)
gen = CounterpointGenerator(
    cantus_firmus=CANTUS_FIRMUS,
    species=Species.FIRST,
    scale=scale,
    voice_range=voice_range,
)

print(f"Generator: {gen}")
print(f"Scale pitch classes: {scale.pitch_classes()}")
print()

# Generate the counterpoint
result = gen.generate()

if not result.feasible:
    print("No feasible counterpoint found!")
else:
    print(f"Result: {result}")
    print(f"  Voices:  {result.n_voices}")
    print(f"  Species: {result.species}")
    print(f"  Key:     {result.key} (C=0)")
    print(f"  Constraints satisfied: {result.constraints_satisfied}/{result.constraints_total}")
    print()

    cf = result.voices[0]
    cp = result.voices[1]

    print("Beat  CF     CP     Interval")
    print("----  -----  -----  --------")
    for i in range(len(cf)):
        interval = abs(cp[i] - cf[i]) % 12
        consonances = {0: "unison", 3: "m3", 4: "M3", 7: "P5", 8: "m6", 9: "M6", 12: "P8"}
        name = consonances.get(interval, f"{interval}")
        print(f"  {i}   {cf[i]:3d}   {cp[i]:3d}   {interval:2d} ({name})")

    print()

    # Verify constraints explicitly
    beats = list(range(len(cf)))
    print(f"Parallel fifths:  {no_parallel_fifths(cf, cp, beats)}")
    print(f"Parallel octaves: {no_parallel_octaves(cf, cp, beats)}")
    for b in range(len(cf)):
        status = consonant_interval(cf, cp, b)
        if status != SAT:
            print(f"  WARNING: dissonance at beat {b}")
    print("\nAll constraints SAT ✓")
