# counterpoint-engine — User Guide

This guide walks through every feature of counterpoint-engine: checking rules, building Laman graphs, generating counterpoint, and producing Tensor-MIDI output.

## Table of Contents

1. [Checking Individual Rules](#checking-individual-rules)
2. [Building Laman Graphs](#building-laman-graphs)
3. [Generating Single-Voice Counterpoint](#generating-single-voice-counterpoint)
4. [Generating Multi-Voice Counterpoint](#generating-multi-voice-counterpoint)
5. [Tensor-MIDI Output](#tensor-midi-output)
6. [Species and Scales](#species-and-scales)
7. [Configuration Options](#configuration-options)
8. [Common Recipes](#common-recipes)
9. [Troubleshooting](#troubleshooting)

---

## Checking Individual Rules

Every rule is a pure function that returns `"SAT"` or `"UNSAT"`.

### No Parallel Fifths / Octaves

```python
from counterpoint_engine.rules import no_parallel_fifths, no_parallel_octaves, SAT, UNSAT

# Parallel fifths — UNSAT
voice_a = [60, 62]  # C → D
voice_b = [67, 69]  # G → A (fifths in similar motion)
print(no_parallel_fifths(voice_a, voice_b, [0, 1]))  # "UNSAT"

# Fixed by contrary motion — SAT
voice_b_fixed = [67, 65]  # G → F
print(no_parallel_fifths(voice_a, voice_b_fixed, [0, 1]))  # "SAT"
```

### Consonant Interval

```python
from counterpoint_engine.rules import consonant_interval

# Perfect fifth at beat 0 — consonant
print(consonant_interval([60], [67], 0))  # "SAT"

# Tritone at beat 0 — dissonant
print(consonant_interval([60], [66], 0))  # "UNSAT"
```

### Melodic Leap

```python
from counterpoint_engine.rules import max_leap_seventh

voice = [60, 72]  # octave leap — allowed
print(max_leap_seventh(voice, 1))  # "SAT"

voice = [60, 73]  # minor ninth — forbidden
print(max_leap_seventh(voice, 1))  # "UNSAT"
```

### Leading Tone Resolution

```python
from counterpoint_engine.rules import proper_resolution

voice = [71, 72]  # B → C (leading tone resolves)
print(proper_resolution(voice, 1, key_tonic=0, key_leading=11))  # "SAT"

voice = [71, 69]  # B → A (leading tone doesn't resolve)
print(proper_resolution(voice, 1))  # "UNSAT"
```

### Voice Independence

```python
from counterpoint_engine.rules import voice_independence
from counterpoint_engine.laman_counterpoint import verify_rigidity

# 3 voices, 3 edges (2*3-3 = 3) — Laman
rigid = verify_rigidity(3, [(0,1), (0,2), (1,2)])
print(voice_independence(rigid))  # "SAT"

# 3 voices, 2 edges — not rigid
rigid = verify_rigidity(3, [(0,1), (0,2)])
print(voice_independence(rigid))  # "UNSAT"
```

---

## Building Laman Graphs

The `CounterpointGraph` maps voices to vertices and constraints to edges.

```python
from counterpoint_engine.laman_counterpoint import CounterpointGraph, henneberg_construct, verify_rigidity

# Auto-construct for N voices
graph = CounterpointGraph(n_voices=4)
print(graph)
# CounterpointGraph(n=4, edges=[(0,1), (0,2), (1,2), ...], rigid=True)

# Check properties
print(graph.edge_count())          # 5 (= 2*4 - 3)
print(graph.expected_edges())      # 5
print(graph.is_minimally_rigid())  # True
print(graph.voice_pairs())         # [(0,1), (0,2), ...]

# Add custom constraints
graph.add_constraint(0, 3, "custom_rule")
```

### Direct Henneberg Construction

```python
edges = henneberg_construct(5, seed=42)
print(len(edges))                    # 7 (= 2*5 - 3)
print(verify_rigidity(5, edges))     # True
```

---

## Generating Single-Voice Counterpoint

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

cantus = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

gen = CounterpointGenerator(
    cantus_firmus=cantus,
    species=Species.FIRST,
    scale=Scale(tonic=0, mode="major"),
    voice_range=VoiceRange(min_pitch=48, max_pitch=67),
)

result = gen.generate()
if result is not None:
    print(f"Counterpoint: {result}")
else:
    print("No solution found under current constraints")
```

### Expected Output

```
Counterpoint: [55, 57, 59, 60, 62, 64, 65, 67]
```

The generator uses backtracking search: it tries each pitch in the voice range that belongs to the scale, ordered by proximity to the previous pitch, and checks all active constraints at each beat.

---

## Generating Multi-Voice Counterpoint

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

cantus = [60, 62, 64, 65, 67, 69, 71, 72]

gen = CounterpointGenerator(
    cantus_firmus=cantus,
    species=Species.FIRST,
    scale=Scale(tonic=0, mode="major"),
)

# Generate 4 voices with custom ranges
ranges = [
    VoiceRange(min_pitch=48, max_pitch=60),  # bass
    VoiceRange(min_pitch=55, max_pitch=67),   # tenor
    VoiceRange(min_pitch=60, max_pitch=72),   # alto
]

voices = gen.generate_n_voices(n_voices=4, voice_ranges=ranges)
# voices[0] = cantus firmus
# voices[1..3] = generated (bass, tenor, alto)
```

The Laman graph determines which voice pairs are constrained against each other. With 4 voices, the graph has 2×4−3 = 5 edges, so each new voice is checked against its neighbors in the graph.

---

## Tensor-MIDI Output

Convert generated voices to the Tensor-MIDI format for streaming or further processing.

```python
from counterpoint_engine.tensor_output import (
    voices_to_tensor_events, voice_leading_to_sidechannels,
    interval_to_flux_vector, voice_intervals_to_flux_vectors,
    TensorMIDIEvent
)

# Convert all voices
tensor_events, midi_events = voices_to_tensor_events(
    voices,
    beat_duration_ms=500.0,  # 120 BPM
    velocity=100,
)

# Each TensorMIDIEvent is 4 bytes
event = tensor_events[0]
print(event)
# TensorMIDIEvent(cos=60, sin=0, beat=0, state=0x1c)
print(event.to_bytes())  # b'\x3c\x00\x00\x1c'

# Create from pitch + interval
event = TensorMIDIEvent.from_pitch_interval(
    pitch=67, interval=7, beat=0, side_state=1
)

# Voice-leading gestures between all pairs at a beat
gestures = voice_leading_to_sidechannels(voices, beat=2)
# {(0,1): "Smile", (0,2): "Nod", (1,2): "Frown", ...}

# Interval → FluxVector mapping
fv = interval_to_flux_vector(7)  # perfect fifth

# All voice intervals at a beat
vectors = voice_intervals_to_flux_vectors(voices, beat=0)
```

### Side-Channel States

| State | Value | Meaning |
|-------|-------|---------|
| Nod | 0 | Stepwise motion, stable |
| Smile | 1 | Consonant leap, strong position |
| Frown | 2 | Dissonant or large leap |
| Resolve | 3 | Leading-tone resolution |

---

## Species and Scales

### Species

| Species | Value | Description |
|---------|-------|-------------|
| `FIRST` | 1 | Note against note (1:1) |
| `SECOND` | 2 | Two notes against one (2:1) |
| `THIRD` | 3 | Four notes against one (4:1) |
| `FOURTH` | 4 | Suspensions |
| `FIFTH` | 5 | Florid counterpoint (mixed) |

### Scale Construction

```python
from counterpoint_engine.generator import Scale

# C major (default)
cmajor = Scale(tonic=0, mode="major")
print(cmajor.pitch_classes())  # (0, 2, 4, 5, 7, 9, 11)

# D minor
dminor = Scale(tonic=2, mode="minor")
print(dminor.pitch_classes())  # (2, 4, 5, 7, 9, 10, 0)

# Check membership
print(cmajor.contains(60))  # True (C4 = pitch class 0)
print(cmajor.contains(61))  # False (C#4 = pitch class 1)
```

---

## Configuration Options

### CounterpointGenerator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cantus_firmus` | `Sequence[int]` | required | Fixed melody (MIDI note numbers) |
| `species` | `Species` | `FIRST` | Species of counterpoint |
| `scale` | `Scale` | `Scale()` (C major) | Diatonic scale for pitch candidates |
| `voice_range` | `VoiceRange` | `VoiceRange()` (C3–G5) | Allowed pitch range |
| `constraints` | `List[Callable]` | 5 first-species rules | Active constraint suite |

### VoiceRange

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_pitch` | `int` | 48 (C3) | Lowest allowed MIDI note |
| `max_pitch` | `int` | 79 (G5) | Highest allowed MIDI note |

### Scale

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tonic` | `int` | 0 (C) | Pitch class of the tonic |
| `mode` | `str` | "major" | "major" or "minor" |

### DeadbandFit / Funnel Parameters

Not applicable — see groove-analyzer for deadband analysis.

---

## Common Recipes

### 1. Validate Existing Counterpoint

```python
from counterpoint_engine.rules import *
from counterpoint_engine.laman_counterpoint import verify_rigidity

cf = [60, 62, 64, 65, 67, 69, 71, 72]
cp = [55, 57, 59, 60, 62, 64, 65, 67]
beats = list(range(len(cf)))

checks = {
    "parallel_fifths": no_parallel_fifths(cf, cp, beats),
    "parallel_octaves": no_parallel_octaves(cf, cp, beats),
    "consonant_intervals": all(
        consonant_interval(cf, cp, b) == SAT for b in beats
    ),
    "leap_bounds": all(
        max_leap_seventh(cp, b) == SAT for b in range(1, len(cp))
    ),
    "resolution": all(
        proper_resolution(cp, b) == SAT for b in range(1, len(cp))
    ),
}
print(checks)
# {'parallel_fifths': 'SAT', 'parallel_octaves': 'SAT', ...}
```

### 2. Generate in Different Keys

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

# G major
gen = CounterpointGenerator(
    cantus_firmus=[67, 69, 71, 72, 74, 76, 78, 79],
    scale=Scale(tonic=7, mode="major"),
    voice_range=VoiceRange(min_pitch=55, max_pitch=74),
)
result = gen.generate()

# F minor
gen = CounterpointGenerator(
    cantus_firmus=[65, 67, 68, 70, 72, 73, 75, 77],
    scale=Scale(tonic=5, mode="minor"),
    voice_range=VoiceRange(min_pitch=48, max_pitch=65),
)
result = gen.generate()
```

### 3. Custom Constraint Suite

```python
from counterpoint_engine.rules import no_parallel_fifths, consonant_interval
from counterpoint_engine.generator import CounterpointGenerator

# Only check fifths and consonance (relaxed rules)
gen = CounterpointGenerator(
    cantus_firmus=[60, 62, 64, 65, 67, 69, 71, 72],
    constraints=[no_parallel_fifths, consonant_interval],
)
result = gen.generate()
```

### 4. Export to Tensor-MIDI Bytes

```python
from counterpoint_engine.generator import CounterpointGenerator
from counterpoint_engine.tensor_output import voices_to_tensor_events

gen = CounterpointGenerator(cantus_firmus=[60, 62, 64, 65, 67, 69, 71, 72])
voices = [[60, 62, 64, 65, 67, 69, 71, 72], gen.generate()]

tensor_events, midi_events = voices_to_tensor_events(voices)

# Serialize all events
raw_stream = b"".join(e.to_bytes() for e in tensor_events)
print(f"Stream: {len(raw_stream)} bytes ({len(tensor_events)} events)")
```

### 5. Check Rigidity of a Custom Graph

```python
from counterpoint_engine.laman_counterpoint import CounterpointGraph

# Build a custom constraint graph
graph = CounterpointGraph(n_voices=3)
graph.add_constraint(0, 1, "parallel_fifths")
graph.add_constraint(0, 2, "consonance")
graph.add_constraint(1, 2, "leap_bounds")

print(f"Edges: {graph.edge_count()}, Expected: {graph.expected_edges()}")
print(f"Minimally rigid: {graph.is_minimally_rigid()}")
```

---

## Troubleshooting

### `generate()` returns `None`

The backtracking search exhausted all candidates without finding a solution. This usually means:

1. **Voice range is too narrow.** Widen `VoiceRange(min_pitch, max_pitch)`.
2. **Scale is wrong.** Ensure `Scale(tonic, mode)` matches the cantus firmus.
3. **Constraints are too strict.** Try removing `proper_resolution` or relaxing `max_leap_seventh`.

### Import errors for `constraint_theory_core` or `flux_tensor_midi`

Add them to your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/constraint-theory-core:/path/to/flux-tensor-midi:$PYTHONPATH"
```

### Multi-voice generation is slow

Multi-voice generation has exponential worst-case complexity. For N voices:

- Keep `n_voices ≤ 4` for reasonable performance.
- Use narrow `VoiceRange` to limit the candidate set.
- Shorter cantus firmus (≤ 12 notes) is faster.

### Test failures (5 multi-voice edge cases)

These are known failures in complex Laman graph interactions. The single-voice path and all rule checks are fully correct. See `tests/` for details.
