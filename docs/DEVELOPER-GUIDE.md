# Counterpoint Engine — Developer Guide

## Architecture

```
counterpoint_engine/
├── __init__.py              # Public API re-exports
├── rules.py                 # FLUX constraint kernels (SAT/UNSAT)
├── generator.py             # Backtracking counterpoint generator
├── laman_counterpoint.py    # Laman graph for constraint rigidity
├── tensor_output.py         # Tensor-MIDI event encoding
tests/
├── test_rules.py            # Rule unit tests
├── test_generator.py        # Generator tests
├── test_laman.py            # Laman graph tests
├── test_tensor_output.py    # Tensor output tests
├── test_integration.py      # End-to-end tests
```

### Module Diagram

```
                    ┌─────────────┐
                    │   rules.py  │  SAT/UNSAT constraint functions
                    └──────┬──────┘
                           │ constraints used by
                    ┌──────▼──────┐
                    │ generator.py│  Backtracking search
                    └──────┬──────┘
                           │ voices produced by
              ┌────────────┼────────────┐
              │            │            │
      ┌───────▼──────┐  ┌──▼──────────┐│
      │laman_        │  │tensor_      ││
      │counterpoint.py│  │output.py   ││
      └──────────────┘  └─────────────┘│
              │                        │
      constraint_theory_core    flux_tensor_midi
```

### Data Flow

1. User provides `cantus_firmus` (list of MIDI note numbers)
2. `CounterpointGenerator` builds a candidate counterpoint via backtracking
3. Each candidate is checked against all active rules from `rules.py`
4. `CounterpointGraph` (Laman graph) ensures minimal rigidity for multi-voice
5. Output voices are encoded to `TensorMIDIEvent` via `tensor_output.py`

## Extending

### Adding a New Constraint Rule

All rules are functions that return `"SAT"` or `"UNSAT"`:

```python
# In rules.py or your own module

SAT = "SAT"
UNSAT = "UNSAT"

def no_voice_crossing(
    voice_lower: Sequence[int],
    voice_upper: Sequence[int],
    beats: Sequence[int],
) -> str:
    """Ensure the upper voice stays above the lower voice."""
    for b in beats:
        if voice_upper[b] < voice_lower[b]:
            return UNSAT
    return SAT
```

Then use it in the generator:

```python
from counterpoint_engine import CounterpointGenerator, Species
from my_rules import no_voice_crossing

gen = CounterpointGenerator(
    cantus_firmus=[60, 62, 64, 65, 67],
    constraints=[
        no_parallel_fifths,
        no_parallel_octaves,
        no_voice_crossing,  # added
    ],
)
```

The generator dispatches rules by function name. Rules named `no_parallel_*` receive `(voice_a, voice_b, beats)`. Rules named `proper_resolution` or `max_leap_seventh` receive `(voice, beat)`. For custom dispatch, the fallback tries `(cf, cp, beats)` then `(cp, beat)`.

### Adding a New Species

The `Species` enum currently defines FIRST through FIFTH, but the generator only fully implements first-species logic. To add second-species (2:1):

```python
# Subclass or modify CounterpointGenerator
class SecondSpeciesGenerator(CounterpointGenerator):
    species = Species.SECOND

    def generate(self):
        # Override: generate 2 notes per cantus firmus note
        # Check constraints at both subdivisions
        ...
```

### Adding a New Output Format

Follow the pattern in `tensor_output.py`:

```python
def voices_to_my_format(voices, **kwargs):
    """Convert voices to custom format."""
    events = []
    for beat in range(len(voices[0])):
        for v_idx, voice in enumerate(voices):
            events.append(my_encode(voice[beat], v_idx, beat))
    return events
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific modules
pytest tests/test_rules.py
pytest tests/test_generator.py

# With verbose output
pytest -v

# With coverage
pytest --cov=counterpoint_engine
```

### Adding Tests

Tests use `pytest` with standard assertions:

```python
# tests/test_my_rule.py
from counterpoint_engine.rules import SAT, UNSAT
from counterpoint_engine import no_parallel_fifths

def test_my_rule_sat():
    voice_a = [60, 62, 64]
    voice_b = [69, 67, 65]  # contrary motion
    assert no_parallel_fifths(voice_a, voice_b, [0, 1, 2]) == SAT

def test_my_rule_unsat():
    voice_a = [60, 62]
    voice_b = [67, 69]  # parallel fifths in similar motion
    assert no_parallel_fifths(voice_a, voice_b, [0, 1]) == UNSAT
```

### Test Categories

- **Unit tests** (`test_rules.py`): Each constraint in isolation
- **Generator tests** (`test_generator.py`): Backtracking search correctness
- **Laman tests** (`test_laman.py`): Graph rigidity verification
- **Tensor tests** (`test_tensor_output.py`): Event encoding round-trips
- **Integration tests** (`test_integration.py`): Full pipeline

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-rule`
3. Write tests first
4. Implement the feature
5. Ensure all tests pass: `pytest`
6. Commit with descriptive message
7. Open a PR

### Code Style

- Python 3.10+ syntax (type hints, `from __future__ import annotations`)
- Functions return `SAT` or `UNSAT` strings (not booleans) — this is the FLUX protocol
- Dataclasses for structured data (`CounterpointGraph`, `TensorMIDIEvent`)
- Docstrings on all public functions with Parameters/Returns sections
- Imports from `constraint_theory_core` and `flux_tensor_midi` at module level

### Build System

No `setup.py` — install with `pip install -e .` in development. Dependencies:
- `constraint-theory-core` (provides `is_laman`, `henneberg_construct`, A₂ lattice)
- `flux-tensor-midi` (provides `FluxVector`, `MidiEvent`)
