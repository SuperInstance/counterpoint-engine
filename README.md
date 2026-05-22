# counterpoint-engine

🎵 Species counterpoint as constraint satisfaction — every rule returns SAT/UNSAT, voices form a Laman graph.

## What It Does

Generates multi-voice counterpoint against a cantus firmus using backtracking search over musical constraints, then outputs the result as Tensor-MIDI events. Each contrapuntal rule is a predicate returning `"SAT"` or `"UNSAT"`; each voice is a vertex in a Laman graph; every constraint is an edge.

## Why It Exists

Species counterpoint has been taught as a set of prohibitions for centuries. This library treats those prohibitions as **constraint predicates** and proves that the constraint graph on N voices is a **Laman graph** (2N−3 edges, minimally rigid). That guarantees no voice is redundant and every rule is load-bearing. If you remove any edge, the structure gains a degree of freedom — a voice can drift unconstrained.

The math: a set of N points in the plane is rigid iff the bar-and-joint framework on those points is Laman. Counterpoint voices are the points; interval constraints are the bars.

## Quick Start

```bash
pip install -e .
```

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

# Define a cantus firmus (C major, 8 notes)
cantus = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

gen = CounterpointGenerator(
    cantus_firmus=cantus,
    species=Species.FIRST,
    scale=Scale(tonic=0, mode="major"),
    voice_range=VoiceRange(min_pitch=48, max_pitch=67),
)

counterpoint = gen.generate()
print(counterpoint)
# → [55, 57, 59, 60, 62, 64, 65, 67]

# Multi-voice — Laman graph guarantees independence
voices = gen.generate_n_voices(n_voices=4)
# voices[0] = cantus firmus, voices[1..3] = generated

# Tensor-MIDI output
from counterpoint_engine.tensor_output import voices_to_tensor_events
tensor_events, midi_events = voices_to_tensor_events(voices)
print(tensor_events[0].to_bytes())  # b'\x3c\x00\x00\x0c'
```

## API Overview

### Rules (`counterpoint_engine.rules`)

Every rule returns the string `"SAT"` or `"UNSAT"`.

```python
from counterpoint_engine.rules import (
    no_parallel_fifths, no_parallel_octaves, proper_resolution,
    max_leap_seventh, consonant_interval, voice_independence, SAT, UNSAT
)

voice_a = [60, 62, 64, 65]
voice_b = [67, 69, 67, 69]
beats = [0, 1, 2, 3]

assert no_parallel_fifths(voice_a, voice_b, beats) == SAT
assert consonant_interval(voice_a, voice_b, 0) == SAT
```

| Function | Signature | What it checks |
|----------|-----------|----------------|
| `no_parallel_fifths` | `(voice_a, voice_b, beats) → str` | No consecutive perfect fifths in similar motion |
| `no_parallel_octaves` | `(voice_a, voice_b, beats) → str` | No consecutive perfect octaves in similar motion |
| `proper_resolution` | `(voice, beat, key_tonic, key_leading) → str` | Leading tone resolves to tonic |
| `max_leap_seventh` | `(voice, beat, max_leap) → str` | Melodic leap ≤ minor seventh (10 semitones) |
| `consonant_interval` | `(voice_a, voice_b, beat, allowed) → str` | Interval at beat is a consonance |
| `voice_independence` | `(laman_check: bool) → str` | Constraint graph is Laman rigid |

### Laman Graphs (`counterpoint_engine.laman_counterpoint`)

```python
from counterpoint_engine.laman_counterpoint import (
    CounterpointGraph, henneberg_construct, verify_rigidity
)

graph = CounterpointGraph(n_voices=4)
print(graph.edges)              # [(0,1), (0,2), (1,2), ...]
print(graph.verify_rigidity())  # True
print(graph.edge_count())       # 5 (= 2*4 - 3)
print(graph.is_minimally_rigid())  # True

edges = henneberg_construct(4, seed=42)
assert verify_rigidity(4, edges)
```

| Class/Function | Description |
|----------------|-------------|
| `CounterpointGraph` | Laman graph with `add_constraint()`, `verify_rigidity()`, `is_minimally_rigid()` |
| `henneberg_construct(n, seed)` | Build a Laman graph via Henneberg type-I construction |
| `verify_rigidity(n_voices, edges)` | Check Laman conditions (2N−3 edges + subset condition) |

### Generator (`counterpoint_engine.generator`)

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

gen = CounterpointGenerator(
    cantus_firmus=[60, 62, 64, 65, 67, 69, 71, 72],
    species=Species.FIRST,
    scale=Scale(tonic=0, mode="major"),
    voice_range=VoiceRange(min_pitch=48, max_pitch=72),
)

# Single voice
counterpoint = gen.generate()

# Multi-voice
voices = gen.generate_n_voices(n_voices=4)
```

| Class/Enum | Key attributes |
|------------|----------------|
| `Species` | `FIRST`, `SECOND`, `THIRD`, `FOURTH`, `FIFTH` (IntEnum 1–5) |
| `VoiceRange` | `min_pitch`, `max_pitch`, `candidates(scale, prev_pitch)` |
| `Scale` | `tonic`, `mode` ("major"/"minor"), `contains(pitch)`, `pitch_classes()` |
| `CounterpointGenerator` | `generate()`, `generate_n_voices(n, ranges)` |

### Tensor-MIDI Output (`counterpoint_engine.tensor_output`)

```python
from counterpoint_engine.tensor_output import (
    voices_to_tensor_events, voice_leading_to_sidechannels,
    interval_to_flux_vector, voice_intervals_to_flux_vectors,
    TensorMIDIEvent
)

tensor_events, midi_events = voices_to_tensor_events(voices)
raw = tensor_events[0].to_bytes()  # 4 bytes: cos, sin, beat, state

gestures = voice_leading_to_sidechannels(voices, beat=2)
# {(0,1): "Smile", (0,2): "Nod", (1,2): "Frown"}

fv = interval_to_flux_vector(7)  # perfect fifth → FluxVector
```

| Function | Returns |
|----------|---------|
| `voices_to_tensor_events(voices)` | `(List[TensorMIDIEvent], List[MidiEvent])` |
| `voice_leading_to_sidechannels(voices, beat)` | `Dict[(i,j), str]` — Nod/Smile/Frown |
| `interval_to_flux_vector(interval)` | `FluxVector` via A₂ lattice |
| `voice_intervals_to_flux_vectors(voices, beat)` | `List[FluxVector]` |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   counterpoint-engine                │
│                                                     │
│  rules.py          laman_counterpoint.py             │
│  ┌──────────┐      ┌──────────────────┐             │
│  │ SAT/UNSAT│◄─────│ CounterpointGraph│             │
│  │ kernels  │      │ henneberg_construct│            │
│  └────┬─────┘      └────────┬─────────┘             │
│       │                     │                        │
│       ▼                     ▼                        │
│  generator.py                                         │
│  ┌──────────────────────────────────┐               │
│  │ CounterpointGenerator            │               │
│  │  .generate() → List[int]         │               │
│  │  .generate_n_voices() → voices   │               │
│  └──────────────┬───────────────────┘               │
│                 │                                    │
│                 ▼                                    │
│  tensor_output.py                                    │
│  ┌──────────────────────────────────┐               │
│  │ voices_to_tensor_events()        │               │
│  │ voice_leading_to_sidechannels()  │               │
│  │ interval_to_flux_vector()        │               │
│  └──────────────────────────────────┘               │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Dependencies                                       │
│  constraint-theory-core ─ Laman rigidity, A₂ lattice│
│  flux-tensor-midi ─ FluxVector, MidiEvent types     │
└─────────────────────────────────────────────────────┘
```

Data flow: `cantus firmus → generator (backtracking) → voices → tensor_output → TensorMIDIEvent stream`

## Ecosystem

- **[constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)** — Laman rigidity, A₂ lattice, dodecet directions
- **[flux-tensor-midi](https://github.com/SuperInstance/flux-tensor-midi)** — FluxVector, MidiEvent, tensor-midi event stream
- **[plato-room-musician](https://github.com/SuperInstance/plato-room-musician)** — Music theory room in the PLATO knowledge system

## Requirements

- Python ≥ 3.10
- `constraint-theory-core` (from `../constraint-theory-core`, add to `PYTHONPATH`)
- `flux-tensor-midi` (Tensor-MIDI event types)

## Installation

```bash
# Clone with dependencies
git clone https://github.com/SuperInstance/counterpoint-engine.git
git clone https://github.com/SuperInstance/constraint-theory-core.git
git clone https://github.com/SuperInstance/flux-tensor-midi.git

export PYTHONPATH="/path/to/constraint-theory-core:/path/to/flux-tensor-midi:$PYTHONPATH"

cd counterpoint-engine
pip install -e ".[dev]"
pytest
```

## Status

![Tests](https://img.shields.io/badge/tests-73%2F78-passing-yellow) ![Version](https://img.shields.io/badge/version-0.1.0-blue) ![License](https://img.shields.io/badge/license-Apache%202.0-green)

73 of 78 tests pass. 5 multi-voice edge cases fail (complex Laman graph interactions under backtracking).

## License

Apache 2.0
