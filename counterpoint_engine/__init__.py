"""
counterpoint_engine — Species counterpoint as constraint satisfaction.

Proves the core thesis: counterpoint = constraint satisfaction = Laman rigidity.

Each rule returns SAT or UNSAT.
Each voice is a vertex in a Laman graph.
Each contrapuntal constraint is an edge.
"""

from .rules import (
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
    voice_independence,
    SAT,
    UNSAT,
    Satisfiability,
)
from .laman_counterpoint import (
    CounterpointGraph,
    henneberg_construct,
    verify_rigidity,
)
from .generator import (
    CounterpointGenerator,
    CounterpointResult,
    Species,
    VoiceRange,
    Scale,
)
from .tensor_output import (
    TensorMIDIEvent,
    voices_to_tensor_events,
    voice_leading_to_sidechannels,
)
from .exceptions import (
    CounterpointError,
    ConstraintViolationError,
    ParallelFifthsError,
    ParallelOctavesError,
    VoiceCrossingError,
    RangeViolationError,
    ResolutionError,
    LeapViolationError,
    InvalidInputError,
    GenerationError,
)

__version__ = "0.1.0"

__all__ = [
    # Rules
    "no_parallel_fifths",
    "no_parallel_octaves",
    "proper_resolution",
    "max_leap_seventh",
    "consonant_interval",
    "voice_independence",
    "SAT",
    "UNSAT",
    "Satisfiability",
    # Laman
    "CounterpointGraph",
    "henneberg_construct",
    "verify_rigidity",
    # Generator
    "CounterpointGenerator",
    "CounterpointResult",
    "Species",
    "VoiceRange",
    "Scale",
    # Tensor output
    "TensorMIDIEvent",
    "voices_to_tensor_events",
    "voice_leading_to_sidechannels",
    # Exceptions
    "CounterpointError",
    "ConstraintViolationError",
    "ParallelFifthsError",
    "ParallelOctavesError",
    "VoiceCrossingError",
    "RangeViolationError",
    "ResolutionError",
    "LeapViolationError",
    "InvalidInputError",
    "GenerationError",
    # Meta
    "__version__",
]
