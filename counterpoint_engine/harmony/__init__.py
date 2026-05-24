"""
counterpoint_engine.harmony — Holonomy-based harmony analysis.

Provides chord progression analysis grounded in constraint theory and
holonomy detection: harmonic movement is cycle consistency.
"""

from .tonal_graph import TonalGraph, TransitionDirection
from .cycle_checker import (
    compute_holonomy,
    winding_number,
    classify_progression,
    HolonomyResult,
    ProgressionType,
)
from .analyzer import (
    Chord,
    parse_roman,
    analyze_progression,
    detect_modulations,
    score_stability,
    PROGRESSIONS,
    ProgressionAnalysis,
)
from .counterpoint_bridge import (
    analyze_counterpoint_harmony,
    check_voice_leading_holonomy,
)

# Convenience aliases matching expected public API
HolonomyAnalyzer = ProgressionAnalysis  # type: ignore[misc]
CycleChecker = type("CycleChecker", (), {"compute_holonomy": staticmethod(compute_holonomy), "winding_number": staticmethod(winding_number), "classify_progression": staticmethod(classify_progression)})()


__all__ = [
    # Tonal graph
    "TonalGraph",
    "TransitionDirection",
    # Cycle checker
    "compute_holonomy",
    "winding_number",
    "classify_progression",
    "HolonomyResult",
    "ProgressionType",
    # Analyzer
    "Chord",
    "parse_roman",
    "analyze_progression",
    "detect_modulations",
    "score_stability",
    "PROGRESSIONS",
    # Bridge
    "analyze_counterpoint_harmony",
    "check_voice_leading_holonomy",
]
