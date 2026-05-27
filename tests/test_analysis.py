"""Tests for counterpoint_engine.analysis — Species compliance and voice-leading scoring."""

import pytest
from counterpoint_engine.analysis import (
    VoiceLeadingScore,
    SpeciesComplianceReport,
    CounterpointAnalysis,
    analyze_voice_leading,
    score_counterpoint,
)


# Test melodies
CF_SIMPLE = [60, 62, 64, 65, 67]
CP_GOOD = [72, 69, 67, 69, 72]  # Octave, m6, P5, M6, P8 — all consonant, no unisons
CP_PARALLEL_FIFTHS = [60, 62, 64, 65, 67]  # Exact parallel = octaves actually
CP_UNISON = [60, 62, 64, 65, 67]  # Same = parallel octaves


class TestCounterpointAnalysis:
    def test_basic_analysis(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert isinstance(report, SpeciesComplianceReport)
        assert report.species == 1

    def test_good_counterpoint_feasible(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert report.feasible

    def test_compliance_ratio_range(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD)
        report = a.analyze()
        assert 0.0 <= report.compliance_ratio <= 1.0

    def test_empty_cf_raises(self):
        with pytest.raises(ValueError, match="cantus_firmus must not be empty"):
            CounterpointAnalysis([], [60])

    def test_empty_cp_raises(self):
        with pytest.raises(ValueError, match="counterpoint must not be empty"):
            CounterpointAnalysis([60], [])

    def test_invalid_species_raises(self):
        with pytest.raises(ValueError, match="species must be 1"):
            CounterpointAnalysis([60], [62], species=0)

    def test_species_5_raises(self):
        with pytest.raises(ValueError, match="species must be 1"):
            CounterpointAnalysis([60], [62], species=6)

    def test_rule_results_populated(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert len(report.rule_results) > 0
        # Should have no_parallel_fifths and no_parallel_octaves
        assert "no_parallel_fifths" in report.rule_results
        assert "no_parallel_octaves" in report.rule_results

    def test_consonant_intervals_checked(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        # Should have consonance check for each beat
        for b in range(5):
            assert f"consonant_interval_beat_{b}" in report.rule_results

    def test_leap_checks(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        # Should have leap checks for both voices
        assert "max_leap_cf_beat_1" in report.rule_results
        assert "max_leap_cp_beat_1" in report.rule_results

    def test_resolution_checks(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert "resolution_cf_beat_1" in report.rule_results
        assert "resolution_cp_beat_1" in report.rule_results

    def test_different_length_voices(self):
        """Analysis should handle voices of different lengths."""
        cf = [60, 62, 64]
        cp = [67, 65, 64, 62, 60]
        a = CounterpointAnalysis(cf, cp, species=1)
        report = a.analyze()
        assert report.species == 1

    def test_key_tonic_custom(self):
        a = CounterpointAnalysis([62, 64, 66], [69, 67, 66], species=1, key_tonic=2)
        report = a.analyze()
        assert report.feasible


class TestVoiceLeadingScore:
    def test_good_voice_leading(self):
        vl = analyze_voice_leading(CF_SIMPLE, CP_GOOD)
        assert isinstance(vl, VoiceLeadingScore)
        assert vl.contrary_motion_ratio > 0
        assert vl.consonance_ratio > 0
        assert vl.overall_score > 0

    def test_stepwise_motion(self):
        # All stepwise in both voices
        vl = analyze_voice_leading([60, 62, 64, 65], [67, 65, 64, 62])
        assert vl.stepwise_motion_ratio == 1.0

    def test_parallel_violations_counted(self):
        # Exact same melody = parallel octaves
        vl = analyze_voice_leading([60, 62, 64], [60, 62, 64])
        assert vl.parallel_octaves_violations > 0

    def test_no_violations(self):
        vl = analyze_voice_leading(CF_SIMPLE, CP_GOOD)
        assert vl.parallel_fifths_violations == 0
        assert vl.parallel_octaves_violations == 0
        assert vl.leap_violations == 0

    def test_consonance_ratio(self):
        vl = analyze_voice_leading([60, 62, 64], [67, 65, 64])
        assert vl.consonance_ratio == 1.0

    def test_dissonance_detected(self):
        # Tritone at beat 1
        vl = analyze_voice_leading([60, 62], [66, 68])
        assert vl.consonance_ratio < 1.0

    def test_overall_score_range(self):
        vl = analyze_voice_leading(CF_SIMPLE, CP_GOOD)
        assert 0.0 <= vl.overall_score <= 1.0


class TestScoreCounterpoint:
    def test_returns_float(self):
        s = score_counterpoint(CF_SIMPLE, CP_GOOD)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_good_score(self):
        s = score_counterpoint(CF_SIMPLE, CP_GOOD)
        assert s >= 0.7

    def test_bad_score(self):
        # Same melody = parallel octaves
        s = score_counterpoint([60, 62, 64], [60, 62, 64])
        assert s < 0.9

    def test_species_parameter(self):
        s1 = score_counterpoint(CF_SIMPLE, CP_GOOD, species=1)
        s2 = score_counterpoint(CF_SIMPLE, CP_GOOD, species=2)
        assert isinstance(s1, float)
        assert isinstance(s2, float)


class TestAnalyzeVoiceLeading:
    def test_basic(self):
        vl = analyze_voice_leading([60, 62, 64], [67, 65, 64])
        assert vl.contrary_motion_ratio > 0

    def test_single_note(self):
        vl = analyze_voice_leading([60], [67])
        assert vl.consonance_ratio == 1.0

    def test_contrary_motion(self):
        # CF goes up, CP goes down
        vl = analyze_voice_leading([60, 62, 64], [67, 65, 64])
        assert vl.contrary_motion_ratio > 0


class TestSpeciesSpecificChecks:
    def test_species_2_passing_tones(self):
        """Species 2 should check passing tone validity."""
        cf = [60, 64]
        cp = [67, 65, 64, 62]
        a = CounterpointAnalysis(cf, cp, species=2)
        report = a.analyze()
        # Should have passing tone checks
        passing_keys = [k for k in report.rule_results if k.startswith("passing_tone")]
        assert len(passing_keys) > 0

    def test_species_4_suspension_checks(self):
        """Species 4 should check suspension patterns."""
        cf = [60, 62, 64, 65]
        cp = [67, 67, 65, 64]  # Suspension-like: 67 held, then resolves
        a = CounterpointAnalysis(cf, cp, species=4)
        report = a.analyze()
        susp_keys = [k for k in report.rule_results if k.startswith("suspension")]
        assert len(susp_keys) > 0

    def test_species_3_passing_tones(self):
        """Species 3 should also check passing tones."""
        cf = [60, 64]
        cp = [67, 65, 64, 62, 60, 62, 64, 65]
        a = CounterpointAnalysis(cf, cp, species=3)
        report = a.analyze()
        passing_keys = [k for k in report.rule_results if k.startswith("passing_tone")]
        assert len(passing_keys) > 0


class TestComplianceReport:
    def test_total_checks_positive(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert report.total_checks > 0

    def test_passed_leq_total(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert report.passed_checks <= report.total_checks

    def test_feasible_means_all_pass(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        if report.feasible:
            assert all(report.rule_results.values())

    def test_voice_leading_attached(self):
        a = CounterpointAnalysis(CF_SIMPLE, CP_GOOD, species=1)
        report = a.analyze()
        assert isinstance(report.voice_leading, VoiceLeadingScore)
