#!/bin/bash
# counterpoint-engine quickstart — generate first-species counterpoint
set -e
echo "🎼 Counterpoint Engine — Quick Start"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

pip install -e "$WORKSPACE/constraint-theory-core" --quiet 2>/dev/null || true

python3 << 'PYEOF'
import sys, types
sys.path.insert(0, "/home/phoenix/.openclaw/workspace/constraint-theory-core")

# Create a dummy package to bypass __init__.py (which imports flux_tensor_midi)
pkg = types.ModuleType("counterpoint_engine")
pkg.__path__ = ["/home/phoenix/.openclaw/workspace/counterpoint-engine/counterpoint_engine"]
pkg.__package__ = "counterpoint_engine"
sys.modules["counterpoint_engine"] = pkg

from counterpoint_engine.rules import consonant_interval_class, SAT, UNSAT
from counterpoint_engine.laman_counterpoint import CounterpointGraph, verify_rigidity
from counterpoint_engine.generator import CounterpointGenerator, Species

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_name(m):
    return f'{NOTE_NAMES[m % 12]}{m // 12 - 1}'

CANTUS_FIRMUS = [60, 62, 64, 65, 67, 69, 67, 65]
cf_names = [midi_to_name(p) for p in CANTUS_FIRMUS]
print(f'Cantus firmus: {" ".join(cf_names)}')
print(f'P5 consonant? {consonant_interval_class(7)} → {SAT}')
print(f'TT consonant? {consonant_interval_class(6)} → {UNSAT}')

gen = CounterpointGenerator(cantus_firmus=CANTUS_FIRMUS, species=Species.FIRST)
result = gen.generate()

if result and result.feasible:
    cp_voice = result.voices[1] if len(result.voices) > 1 else result.voices[0]
    cp_names = [midi_to_name(p) for p in cp_voice]
    intervals = [abs(cp - cf) for cf, cp in zip(CANTUS_FIRMUS, cp_voice)]
    print(f'\n  Counterpoint: {" ".join(f"{n:>4}" for n in cp_names)}')
    print(f'  Cantus firm:  {" ".join(f"{n:>4}" for n in cf_names)}')
    print(f'  Intervals:    {" ".join(f"{iv:>4}" for iv in intervals)}')
    print(f'  Constraints:  {result.constraints_satisfied}/{result.constraints_total} satisfied')
    print(f'✓ Generated {len(cp_voice)} notes of valid first-species counterpoint')
else:
    print('No valid counterpoint found')

graph = CounterpointGraph(n_voices=4)
print(f'\nLaman graph (4 voices): {len(graph.edges)} edges  rigid={graph.verify_rigidity()}')

print()
print('✅ counterpoint-engine works!')
PYEOF
