# counterpoint-engine

🎵 **Species counterpoint as constraint satisfaction** — every rule returns SAT/UNSAT, voices form a Laman graph, output as Tensor-MIDI events.

## What This Gives You

- **SAT/UNSAT rules** — every contrapuntal prohibition is a constraint predicate
- **Laman graph proof** — N voices with 2N−3 constraints are minimally rigid
- **Multi-voice generation** — backtracking search with constraint propagation
- **Tensor-MIDI output** — voices export as Tensor-MIDI byte events
- **Canon support** — generate canons at configurable intervals
- **Analysis tools** — interval analysis, voice-leading metrics, rule checking

## Installation

```bash
pip install counterpoint-engine
```

## Quick Start

```python
from counterpoint_engine.generator import CounterpointGenerator, Species, Scale, VoiceRange

cantus = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

gen = CounterpointGenerator(
    cantus_firmus=cantus,
    species=Species.FIRST,
    scale=Scale(tonic=0, mode="major"),
    voice_range=VoiceRange(min_pitch=48, max_pitch=67),
)

counterpoint = gen.generate()
# → [48, 53, 52, 50, 48, 48, 50, 48]

# Multi-voice with Laman rigidity proof
voices = gen.generate_n_voices(n_voices=4)
```

## The Math

A set of N points in the plane is rigid iff the bar-and-joint framework is **Laman** (2N−3 edges, no redundant constraints). Counterpoint voices are points; interval constraints are bars. Remove any edge → a voice drifts unconstrained.

## Testing

```bash
pip install -e .
pytest
```

## How It Fits

Music theory engine in the SuperInstance ecosystem. Feeds `symplectic-music` (Hamiltonian harmony), `holonomy-harmony-rs` (holonomy analysis), and `superinstance-live` (DAW integration).

## License

MIT
