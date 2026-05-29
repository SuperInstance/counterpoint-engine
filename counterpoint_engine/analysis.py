"""Counterpoint analysis — score species compliance and voice-leading quality.

This module provides analytical tools that evaluate an existing pair of
voices (or multi-voice texture) against classical counterpoint rules and
report numeric scores.  It is **read-only**: it does not generate or modify
any music, only grades what it receives.

Key classes:

- :class:`VoiceLeadingScore` — breakdown of voice-leading quality metrics.
- :class:`SpeciesComplianceReport` — detailed per-rule compliance report.
- :class:`CounterpointAnalysis` — main entry point: build once, call
  :meth:`~CounterpointAnalysis.analyze` to get a full report.

No external dependencies beyond the standard library and ``counterpoint_engine``
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from counterpoint_engine.rules import (
    SAT,
    consonant_interval_class,
    contrary_motion_score,
    is_step,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
    _motion_type,
)


# ---------------------------------------------------------------------------
# Scoring data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VoiceLeadingScore:
    """Quantitative assessment of voice-leading quality between two voices.

    All scores are in the range 0.0–1.0 (higher is better).

    Attributes
    ----------
    contrary_motion_ratio : float
        Fraction of beats with contrary motion.
    stepwise_motion_ratio : float
        Fraction of melodic intervals that are steps (1–2 semitones).
    consonance_ratio : float
        Fraction of beats where the harmonic interval is consonant.
    parallel_fifths_violations : int
        Number of beat-pairs with parallel fifths.
    parallel_octaves_violations : int
        Number of beat-pairs with parallel octaves.
    leap_violations : int
        Number of beats where the melodic leap exceeds a minor seventh.
    resolution_violations : int
        Number of leading-tone mis-resolutions.
    overall_score : float
        Weighted composite of the above metrics.
    """

    contrary_motion_ratio: float
    stepwise_motion_ratio: float
    consonance_ratio: float
    parallel_fifths_violations: int
    parallel_octaves_violations: int
    leap_violations: int
    resolution_violations: int
    overall_score: float


@dataclass(frozen=True, slots=True)
class SpeciesComplianceReport:
    """Detailed compliance report for a species counterpoint exercise.

    Attributes
    ----------
    species : int
        Species number (1–5).
    total_checks : int
        Total number of individual rule checks performed.
    passed_checks : int
        Number of checks that passed.
    compliance_ratio : float
        ``passed_checks / total_checks`` (1.0 = perfect).
    rule_results : Dict[str, bool]
        Per-rule pass/fail (True = SAT).
    voice_leading : VoiceLeadingScore
        Detailed voice-leading quality metrics.
    feasible : bool
        True if **all** hard constraints are satisfied.
    """

    species: int
    total_checks: int
    passed_checks: int
    compliance_ratio: float
    rule_results: Dict[str, bool]
    voice_leading: VoiceLeadingScore
    feasible: bool


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

@dataclass
class CounterpointAnalysis:
    """Analyze an existing counterpoint texture for species compliance.

    This is the main entry point.  Construct with the voices and species,
    then call :meth:`analyze`.

    Parameters
    ----------
    cantus_firmus : Sequence[int]
        The fixed voice (index 0).
    counterpoint : Sequence[int]
        The generated voice (index 1).
    species : int
        Species number (1–5).
    key_tonic : int
        Tonic pitch class (default 0 = C).

    Example
    -------
    >>> cf = [60, 62, 64, 65, 67]
    >>> cp = [67, 65, 64, 62, 60]
    >>> analysis = CounterpointAnalysis(cf, cp, species=1)
    >>> report = analysis.analyze()
    >>> report.feasible
    True
    """

    cantus_firmus: Sequence[int]
    counterpoint: Sequence[int]
    species: int = 1
    key_tonic: int = 0

    def __post_init__(self) -> None:
        if not self.cantus_firmus:
            raise ValueError("cantus_firmus must not be empty")
        if not self.counterpoint:
            raise ValueError("counterpoint must not be empty")
        if self.species < 1 or self.species > 5:
            raise ValueError(f"species must be 1–5, got {self.species}")

    def analyze(self) -> SpeciesComplianceReport:
        """Run a full analysis and return a compliance report.

        Returns
        -------
        SpeciesComplianceReport
            Detailed report with per-rule results and voice-leading score.
        """
        rule_results: Dict[str, bool] = {}
        total = 0
        passed = 0

        beats = list(range(min(len(self.cantus_firmus), len(self.counterpoint))))

        # --- Hard constraints ---

        # 1. No parallel fifths
        if len(beats) >= 2:
            total += 1
            result = no_parallel_fifths(
                self.cantus_firmus, self.counterpoint, beats
            )
            rule_results["no_parallel_fifths"] = result == SAT
            if result == SAT:
                passed += 1

        # 2. No parallel octaves
        if len(beats) >= 2:
            total += 1
            result = no_parallel_octaves(
                self.cantus_firmus, self.counterpoint, beats
            )
            rule_results["no_parallel_octaves"] = result == SAT
            if result == SAT:
                passed += 1

        # 3. Consonant intervals (species-dependent)
        for b in beats:
            total += 1
            result = consonant_interval(
                self.cantus_firmus,
                self.counterpoint,
                b,
                total_beats=len(beats),
            )
            rule_results[f"consonant_interval_beat_{b}"] = result == SAT
            if result == SAT:
                passed += 1

        # 4. Max leap (applied to both voices)
        for voice_name, voice in [("cf", self.cantus_firmus), ("cp", self.counterpoint)]:
            for b in range(1, len(voice)):
                total += 1
                result = max_leap_seventh(voice, b)
                rule_results[f"max_leap_{voice_name}_beat_{b}"] = result == SAT
                if result == SAT:
                    passed += 1

        # 5. Proper resolution (leading tone → tonic)
        for voice_name, voice in [("cf", self.cantus_firmus), ("cp", self.counterpoint)]:
            for b in range(1, len(voice)):
                total += 1
                result = proper_resolution(voice, b)
                rule_results[f"resolution_{voice_name}_beat_{b}"] = result == SAT
                if result == SAT:
                    passed += 1

        # --- Species-specific soft checks ---
        if self.species in (2, 3):
            self._check_passing_tones(beats, rule_results)
        if self.species == 4:
            self._check_suspensions(beats, rule_results)

        # --- Voice leading quality ---
        vl = self._voice_leading_score(beats)

        compliance = passed / total if total > 0 else 1.0
        feasible = all(rule_results.values())

        return SpeciesComplianceReport(
            species=self.species,
            total_checks=total,
            passed_checks=passed,
            compliance_ratio=compliance,
            rule_results=rule_results,
            voice_leading=vl,
            feasible=feasible,
        )

    def _voice_leading_score(self, beats: List[int]) -> VoiceLeadingScore:
        """Compute voice-leading quality metrics.

        Parameters
        ----------
        beats : List[int]
            Beat indices.

        Returns
        -------
        VoiceLeadingScore
            Detailed breakdown.
        """
        cf = self.cantus_firmus
        cp = self.counterpoint

        # Contrary motion ratio
        contrary = contrary_motion_score(cf, cp, beats)

        # Stepwise motion ratio (for both voices)
        total_melodic = 0
        stepwise = 0
        for voice in (cf, cp):
            for b in range(1, len(voice)):
                total_melodic += 1
                if is_step(voice[b - 1], voice[b]):
                    stepwise += 1
        step_ratio = stepwise / total_melodic if total_melodic > 0 else 1.0

        # Consonance ratio
        consonant_count = 0
        for b in beats:
            intv = abs(cf[b] - cp[b]) % 12
            if consonant_interval_class(intv):
                consonant_count += 1
        cons_ratio = consonant_count / len(beats) if beats else 1.0

        # Count parallel violations
        pf_violations = 0
        po_violations = 0
        for i in range(1, len(beats)):
            prev = beats[i - 1]
            curr = beats[i]
            int_prev = abs(cf[prev] - cp[prev]) % 12
            int_curr = abs(cf[curr] - cp[curr]) % 12
            motion = _motion_type(cf, cp, curr)
            if int_prev == 7 and int_curr == 7 and motion in ("similar", "static"):
                pf_violations += 1
            if int_prev == 0 and int_curr == 0 and motion in ("similar", "static"):
                po_violations += 1

        # Count leap violations
        leap_viol = 0
        for voice in (cf, cp):
            for b in range(1, len(voice)):
                leap = abs(voice[b] - voice[b - 1])
                simple = leap % 12
                if simple > 10 and leap > 10 and simple != 12:
                    leap_viol += 1

        # Count resolution violations
        res_viol = 0
        key_leading = (self.key_tonic + 11) % 12
        for voice in (cf, cp):
            for b in range(1, len(voice)):
                prev_pc = voice[b - 1] % 12
                curr_pc = voice[b] % 12
                if prev_pc == key_leading and curr_pc != self.key_tonic:
                    res_viol += 1

        # Overall: weighted composite
        # Hard constraints matter more than soft
        overall = (
            contrary * 0.25
            + step_ratio * 0.15
            + cons_ratio * 0.30
            + (1.0 - min(pf_violations / max(len(beats), 1), 1.0)) * 0.15
            + (1.0 - min(po_violations / max(len(beats), 1), 1.0)) * 0.15
        )

        return VoiceLeadingScore(
            contrary_motion_ratio=contrary,
            stepwise_motion_ratio=step_ratio,
            consonance_ratio=cons_ratio,
            parallel_fifths_violations=pf_violations,
            parallel_octaves_violations=po_violations,
            leap_violations=leap_viol,
            resolution_violations=res_viol,
            overall_score=overall,
        )

    def _check_passing_tones(
        self,
        beats: List[int],
        rule_results: Dict[str, bool],
    ) -> None:
        """Check that dissonant notes in species 2/3 are valid passing tones.

        A passing tone must be approached and left by step in the same
        direction.

        Parameters
        ----------
        beats : List[int]
            Beat indices.
        rule_results : Dict[str, bool]
            Mutated in-place with results.
        """
        cp = self.counterpoint
        for b in range(1, len(cp) - 1):
            intv = abs(self.cantus_firmus[min(b, len(self.cantus_firmus) - 1)] - cp[b]) % 12
            if not consonant_interval_class(intv):
                # Dissonant note — must be a passing tone
                direction_in = cp[b] - cp[b - 1]
                direction_out = cp[b + 1] - cp[b]
                is_passing = (
                    is_step(cp[b - 1], cp[b])
                    and is_step(cp[b], cp[b + 1])
                    and (
                        (direction_in > 0 and direction_out > 0)
                        or (direction_in < 0 and direction_out < 0)
                    )
                )
                rule_results[f"passing_tone_beat_{b}"] = is_passing

    def _check_suspensions(
        self,
        beats: List[int],
        rule_results: Dict[str, bool],
    ) -> None:
        """Check suspension patterns for species 4.

        A suspension requires:
        1. Preparation: consonant note on the preceding beat.
        2. Suspension: same pitch held (or tied) into a strong beat
           where it becomes dissonant.
        3. Resolution: step downward to a consonance.

        Parameters
        ----------
        beats : List[int]
            Beat indices.
        rule_results : Dict[str, bool]
            Mutated in-place with results.
        """
        cf = self.cantus_firmus
        cp = self.counterpoint
        for b in range(1, len(cp) - 1):
            intv_prev = abs(cf[min(b - 1, len(cf) - 1)] - cp[b - 1]) % 12
            intv_curr = abs(cf[min(b, len(cf) - 1)] - cp[b]) % 12
            intv_next = abs(cf[min(b + 1, len(cf) - 1)] - cp[b + 1]) % 12

            # If current beat is dissonant but prev is consonant
            if not consonant_interval_class(intv_curr) and consonant_interval_class(intv_prev):
                # Check resolution: step down to consonance
                resolved = (
                    is_step(cp[b], cp[b + 1])
                    and cp[b + 1] < cp[b]
                    and consonant_interval_class(intv_next)
                )
                rule_results[f"suspension_beat_{b}"] = resolved


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def analyze_voice_leading(
    voice_a: Sequence[int],
    voice_b: Sequence[int],
) -> VoiceLeadingScore:
    """Quick voice-leading analysis between two voices.

    Parameters
    ----------
    voice_a, voice_b : Sequence[int]
        Two voices to compare.

    Returns
    -------
    VoiceLeadingScore
        Detailed metrics.

    Example
    -------
    >>> vl = analyze_voice_leading([60, 62, 64], [67, 65, 64])
    >>> vl.contrary_motion_ratio > 0
    True
    """
    analysis = CounterpointAnalysis(
        cantus_firmus=voice_a,
        counterpoint=voice_b,
    )
    return analysis._voice_leading_score(list(range(min(len(voice_a), len(voice_b)))))


def score_counterpoint(
    cantus_firmus: Sequence[int],
    counterpoint: Sequence[int],
    species: int = 1,
) -> float:
    """Return a single 0.0–1.0 score for a counterpoint exercise.

    This is the fastest way to get an overall quality metric.

    Parameters
    ----------
    cantus_firmus : Sequence[int]
        The fixed voice.
    counterpoint : Sequence[int]
        The generated voice.
    species : int
        Species number (1–5).

    Returns
    -------
    float
        Overall score in 0.0–1.0 range.

    Example
    -------
    >>> score_counterpoint([60, 62, 64], [67, 65, 64])
    1.0
    """
    analysis = CounterpointAnalysis(
        cantus_firmus=cantus_firmus,
        counterpoint=counterpoint,
        species=species,
    )
    report = analysis.analyze()
    return report.voice_leading.overall_score
