"""
Analyzer — analyze real chord progressions with holonomy theory.

Parses chord symbols, builds a tonal graph, computes holonomy at each
transition, detects modulations and modal interchange, and scores the
progression on a stability / adventurousness axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .tonal_graph import (
    TonalGraph,
    TransitionDirection,
    pc_from_name,
    pitch_name,
    semitone_interval,
)
from .cycle_checker import compute_holonomy, HolonomyResult, ProgressionType


# ---------------------------------------------------------------------------
# Chord dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Chord:
    """
    A chord in a tonal context.

    Attributes
    ----------
    root : int
        Pitch class of the root (0-11).
    quality : str
        Chord quality, e.g. 'maj', 'min', '7', 'maj7', 'dim'.
    function : str
        Roman-numeral function in the home key, e.g. 'I', 'V', 'vi', 'bVI'.
    key : Tuple[int, str]
        (tonic_pc, mode) where mode is 'major' or 'minor'.
    is_diatonic : bool
        Whether the chord belongs to the home key.
    is_secondary_dominant : bool
        Whether this is a V/x chord (tonicization).
    implied_key : Tuple[int, str]
        The key this chord temporarily implies.
    """
    root: int
    quality: str
    function: str
    key: Tuple[int, str]
    is_diatonic: bool
    is_secondary_dominant: bool
    implied_key: Tuple[int, str]

    @property
    def root_name(self) -> str:
        return pitch_name(self.root)

    @property
    def key_name(self) -> str:
        tonic, mode = self.key
        return f"{pitch_name(tonic)} {mode}"

    def __repr__(self) -> str:
        return f"<Chord {self.root_name}{self.quality} ({self.function}) in {self.key_name}>"


# ---------------------------------------------------------------------------
# Roman numeral parsing
# ---------------------------------------------------------------------------

# Major key scale degrees (pitch classes relative to tonic)
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
# Minor key scale degrees (natural minor)
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# Qualities for each scale degree in major
_MAJOR_QUALITIES = ["maj", "min", "min", "maj", "maj", "min", "dim"]
# Qualities in minor
_MINOR_QUALITIES = ["min", "dim", "maj", "min", "min", "maj", "maj"]


def _parse_roman(roman: str) -> Tuple[str, bool, Optional[str]]:
    """
    Parse a Roman numeral string.

    Returns
    -------
    (numeral, is_upper, suffix)
    """
    roman = roman.strip()
    # Detect flat or sharp prefix
    accidental = ""
    if roman.startswith("bb") or roman.startswith("##"):
        accidental = roman[:2]
        roman = roman[2:]
    elif roman.startswith("b"):
        accidental = "b"
        roman = roman[1:]
    elif roman.startswith("#"):
        accidental = "#"
        roman = roman[1:]

    # Split numeral from quality suffix
    match = re.match(r"^(IV|VI{0,3}|I{1,3}|iv|vi{0,3}|i{1,3})(.*)$", roman)
    if not match:
        raise ValueError(f"Cannot parse Roman numeral: {roman}")

    numeral = accidental + match.group(1)
    suffix = match.group(2) or None
    is_upper = match.group(1)[0].isupper()
    return numeral, is_upper, suffix


def _numeral_to_degree(numeral: str) -> int:
    """Convert a Roman numeral to a scale degree (0-indexed)."""
    # Strip accidentals for lookup
    n = numeral.lstrip('b#')
    table = {
        "I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6,
        "i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6,
    }
    return table[n]


def _degree_to_pc(degree: int, key_tonic: int, mode: str, accidental: str = "") -> int:
    """Convert a scale degree to a pitch class."""
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    pc = (key_tonic + scale[degree]) % 12
    if accidental == "bb":
        pc = (pc - 2) % 12
    elif accidental == "##":
        pc = (pc + 2) % 12
    elif accidental == "b":
        pc = (pc - 1) % 12
    elif accidental == "#":
        pc = (pc + 1) % 12
    return pc


def parse_roman(
    symbol: str,
    key_tonic: int = 0,
    mode: str = "major",
) -> Chord:
    """
    Parse a Roman-numeral chord symbol into a Chord object.

    Parameters
    ----------
    symbol : str
        Roman numeral, e.g. 'I', 'V', 'vi', 'bVI', 'V7', 'V/vi'.
    key_tonic : int
        Pitch class of the key tonic (0=C).
    mode : str
        'major' or 'minor'.

    Returns
    -------
    Chord
    """
    symbol = symbol.strip()
    mode = mode.lower()

    # Handle secondary dominants: V/vi, V7/IV, etc.
    if "/" in symbol and not symbol.startswith("b") and not symbol.startswith("#"):
        parts = symbol.split("/", 1)
        primary = parts[0]
        target = parts[1]
        # The target chord's root becomes the temporary tonic
        target_chord = parse_roman(target, key_tonic, mode)
        target_tonic = target_chord.root
        # Parse the dominant chord in the target key
        dom_chord = parse_roman(primary, target_tonic, "major")
        return Chord(
            root=dom_chord.root,
            quality=dom_chord.quality,
            function=symbol,
            key=(key_tonic, mode),
            is_diatonic=False,
            is_secondary_dominant=True,
            implied_key=(target_tonic, "major" if primary[0].isupper() else "minor"),
        )

    numeral, is_upper, suffix = _parse_roman(symbol)

    # Determine accidental
    accidental = ""
    if numeral.startswith("bb") or numeral.startswith("##"):
        accidental = numeral[:2]
        numeral = numeral[2:]
    elif numeral.startswith("b"):
        accidental = "b"
        numeral = numeral[1:]
    elif numeral.startswith("#"):
        accidental = "#"
        numeral = numeral[1:]

    degree = _numeral_to_degree(numeral)
    root = _degree_to_pc(degree, key_tonic, mode, accidental)

    # Determine quality
    if suffix:
        quality = _suffix_to_quality(suffix, is_upper)
    else:
        qualities = _MAJOR_QUALITIES if mode == "major" else _MINOR_QUALITIES
        diatonic_q = qualities[degree]
        if accidental:
            quality = "maj" if is_upper else "min"
        # Respect case: uppercase = major, lowercase = minor
        # unless diatonic quality already matches (e.g. I=maj, ii=min)
        elif is_upper and diatonic_q not in ("maj", "maj7", "7"):
            quality = "maj"
        elif not is_upper and diatonic_q not in ("min", "dim", "min6", "m7", "m7b5"):
            quality = "min"
        else:
            quality = diatonic_q

    # Check diatonic
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    is_diatonic = (root in [(key_tonic + s) % 12 for s in scale]) and not accidental

    return Chord(
        root=root,
        quality=quality,
        function=symbol,
        key=(key_tonic, mode),
        is_diatonic=is_diatonic,
        is_secondary_dominant=False,
        implied_key=(key_tonic, mode),
    )


def _suffix_to_quality(suffix: str, is_upper: bool) -> str:
    """Map a chord suffix to a quality string."""
    suffix = suffix.strip()
    mapping = {
        "7": "7",
        "maj7": "maj7",
        "m7": "m7",
        "min7": "m7",
        "dim": "dim",
        "dim7": "dim7",
        "ø7": "m7b5",
        "m7b5": "m7b5",
        "+": "aug",
        "aug": "aug",
        "sus4": "sus4",
        "sus2": "sus2",
        "6": "maj6",
        "m6": "min6",
        "9": "dom9",
        "maj9": "maj9",
        "m9": "min9",
        "11": "dom11",
        "13": "dom13",
    }
    if suffix in mapping:
        return mapping[suffix]
    # Heuristic
    if suffix.startswith("m") and not suffix.startswith("maj"):
        return "min"
    if is_upper:
        return "maj"
    return "min"


# ---------------------------------------------------------------------------
# Progression analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProgressionAnalysis:
    """Full analysis of a chord progression."""
    chords: List[Chord]
    holonomy: HolonomyResult
    graph: TonalGraph
    modulations: List[Tuple[int, str]]  # (index, description)
    modal_interchanges: List[Tuple[int, str]]
    stability_score: float  # 0.0 = highly unstable/adventurous, 1.0 = completely stable

    def __repr__(self) -> str:
        return (
            f"<ProgressionAnalysis chords={len(self.chords)} "
            f"holonomy={self.holonomy.holonomy} "
            f"stability={self.stability_score:.2f}>"
        )


def analyze_progression(
    symbols: List[str],
    key_tonic: int = 0,
    mode: str = "major",
    wrap: bool = False,
) -> ProgressionAnalysis:
    """
    Analyze a chord progression given as Roman-numeral symbols.

    Parameters
    ----------
    symbols : List[str]
        Roman numerals, e.g. ['I', 'IV', 'V', 'I'].
    key_tonic : int
        Pitch class of the home key tonic.
    mode : str
        'major' or 'minor'.
    wrap : bool
        If True, treat as a closed cycle.

    Returns
    -------
    ProgressionAnalysis
    """
    chords = [parse_roman(s, key_tonic, mode) for s in symbols]
    roots = [c.root for c in chords]

    # Build tonal graph
    graph = TonalGraph()
    graph.build_from_progression(roots)

    # Compute holonomy
    holonomy = compute_holonomy(roots, wrap=wrap)

    # Detect modulations and modal interchange
    modulations, modal_interchanges = _detect_deviations(chords, holonomy)

    # Stability score
    stability = _score_stability(chords, holonomy, modulations, modal_interchanges)

    return ProgressionAnalysis(
        chords=chords,
        holonomy=holonomy,
        graph=graph,
        modulations=modulations,
        modal_interchanges=modal_interchanges,
        stability_score=stability,
    )


def _detect_deviations(
    chords: List[Chord],
    holonomy: HolonomyResult,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Detect modulation points and modal interchange points."""
    modulations: List[Tuple[int, str]] = []
    interchanges: List[Tuple[int, str]] = []
    cumulative = holonomy.cumulative

    for i, chord in enumerate(chords):
        if chord.is_secondary_dominant:
            modulations.append((i, f"Secondary dominant {chord.function} implies {chord.implied_key}"))
            continue

        if not chord.is_diatonic and i > 0:
            # Check if it's borrowed from parallel mode
            home_tonic, home_mode = chord.key
            if home_mode == "major":
                # In major, bIII, bVI, bVII are common borrowings from parallel minor
                if chord.function in ("bIII", "bVI", "bVII", "iv"):
                    interchanges.append((i, f"Borrowed chord {chord.function} from parallel minor"))
                else:
                    modulations.append((i, f"Non-diatonic chord {chord.function}"))
            else:
                # In minor, III, VI, VII are common borrowings from parallel major
                if chord.function in ("III", "VI", "VII", "IV"):
                    interchanges.append((i, f"Borrowed chord {chord.function} from parallel major"))
                else:
                    modulations.append((i, f"Non-diatonic chord {chord.function}"))

    return modulations, interchanges


def _score_stability(
    chords: List[Chord],
    holonomy: HolonomyResult,
    modulations: List[Tuple[int, str]],
    interchanges: List[Tuple[int, str]],
) -> float:
    """
    Compute a stability score 0.0–1.0.

    1.0 = completely diatonic, zero holonomy, no deviations.
    0.0 = highly chromatic, large holonomy, many modulations.
    """
    if not chords:
        return 0.0

    # Base score from diatonic ratio
    diatonic_count = sum(1 for c in chords if c.is_diatonic and not c.is_secondary_dominant)
    score = diatonic_count / len(chords)

    # Penalize holonomy
    if holonomy.holonomy != 0:
        score *= 0.5
    else:
        score = min(1.0, score + 0.1)

    # Penalize max deviation
    if holonomy.max_deviation > 3:
        score *= 0.7
    elif holonomy.max_deviation > 1:
        score *= 0.9

    # Penalize modulations more than interchanges
    score *= max(0.3, 1.0 - len(modulations) * 0.2)
    score *= max(0.8, 1.0 - len(interchanges) * 0.05)

    return round(max(0.0, min(1.0, score)), 3)


def detect_modulations(analysis: ProgressionAnalysis) -> List[Tuple[int, str]]:
    """Return detected modulation points from an analysis."""
    return analysis.modulations


def score_stability(analysis: ProgressionAnalysis) -> float:
    """Return the stability score from an analysis."""
    return analysis.stability_score


# ---------------------------------------------------------------------------
# Famous progression database
# ---------------------------------------------------------------------------

PROGRESSIONS: Dict[str, Tuple[List[str], int, str]] = {
    # Name -> (roman numerals, key_tonic, mode)
    "pachelbel_canon": (
        ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
        2,  # D major
        "major",
    ),
    "blues_12_bar": (
        ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
        0,  # C major
        "major",
    ),
    "giant_steps": (
        ["I", "V", "IV", "V", "bVII", "ii", "V", "I", "bV", "bVII", "bIII", "V", "I"],
        11,  # B major (starts on Bmaj7)
        "major",
    ),
    "chopin_em_prelude": (
        ["i", "V", "bVI", "i", "iv", "i", "bII", "V", "i"],
        4,  # E minor
        "minor",
    ),
    "axis_progression": (
        ["I", "V", "vi", "IV"],
        0,
        "major",
    ),
    "andulusian_cadence": (
        ["i", "bVII", "bVI", "V"],
        0,
        "minor",
    ),
    "doo_wop": (
        ["I", "vi", "IV", "V"],
        0,
        "major",
    ),
    "sensitive_female": (
        ["vi", "IV", "I", "V"],
        0,
        "major",
    ),
    "rhythm_changes": (
        ["I", "vi", "ii", "V", "I", "vi", "ii", "V", "iii", "VI", "ii", "V", "I"],
        0,
        "major",
    ),
    "bird_changes": (
        ["I", "vi", "ii", "V", "iii", "VI", "II", "V", "I"],
        0,
        "major",
    ),
    "montgomery_ward": (
        ["I", "bVII", "IV", "I"],
        0,
        "major",
    ),
    "plagal_cadence": (
        ["IV", "I"],
        0,
        "major",
    ),
    "perfect_cadence": (
        ["V", "I"],
        0,
        "major",
    ),
    "deceptive_cadence": (
        ["V", "vi"],
        0,
        "major",
    ),
    "coltrane_changes": (
        ["I", "III", "bV", "I", "bIII", "bV", "bbVII", "I"],
        0,
        "major",
    ),
    "minor_ii_v_i": (
        ["ii", "V", "i"],
        0,
        "minor",
    ),
    "take_five": (
        ["i", "bVII", "i", "V"],
        0,
        "minor",
    ),
    "hey_jude": (
        ["I", "bVII", "IV", "I"],
        0,
        "major",
    ),
    "creep": (
        ["I", "III", "IV", "iv"],
        0,
        "major",
    ),
    "autumn_leaves": (
        ["ii", "V", "I", "IV", "bVII", "iii", "vi", "II", "bII", "i", "V", "i"],
        0,
        "minor",
    ),
}
