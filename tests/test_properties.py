"""Property-based tests: generated counterpoint always satisfies declared constraints.

Uses Hypothesis for automatic test case generation.
"""

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from counterpoint_engine.generator import (
    CounterpointGenerator,
    Species,
    VoiceRange,
)
from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    no_parallel_fifths,
    no_parallel_octaves,
    consonant_interval,
    max_leap_seventh,
    proper_resolution,
    consonant_interval_class,
)


# Strategies
midi_pitch = st.integers(min_value=48, max_value=84)
cantus_firmus = st.lists(midi_pitch, min_size=3, max_size=12)
tonic_pc = st.integers(min_value=0, max_value=11)


class TestPropertySpecies1:
    """Property: species 1 output always satisfies all constraints."""

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_always_feasible(self, cf: list) -> None:
        """Any reasonable cantus firmus yields a feasible species 1 solution."""
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_no_parallel_fifths(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        beats = list(range(len(cp)))
        assert no_parallel_fifths(cf, cp, beats) == SAT

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_no_parallel_octaves(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        beats = list(range(len(cp)))
        assert no_parallel_octaves(cf, cp, beats) == SAT

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_all_consonant(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        for b in range(len(cp)):
            assert consonant_interval(cf, cp, b) == SAT

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_max_leap(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        for b in range(len(cp)):
            assert max_leap_seventh(cp, b) == SAT

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_proper_resolution(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        for b in range(len(cp)):
            assert proper_resolution(cp, b) == SAT

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_voice_range_respected(self, cf: list) -> None:
        vr = VoiceRange(min_pitch=48, max_pitch=79)
        gen = CounterpointGenerator(
            cantus_firmus=cf, species=Species.FIRST, voice_range=vr,
        )
        result = gen.generate()
        assume(result.feasible)
        for p in result.voices[1]:
            assert vr.min_pitch <= p <= vr.max_pitch

    @given(cf=cantus_firmus)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_note_count_equals_cf(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assume(result.feasible)
        assert len(result.voices[1]) == len(cf)


class TestPropertySpecies2:
    """Property: species 2 strong beats are consonant and weak beats stepwise."""

    @given(cf=cantus_firmus)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_strong_beats_consonant(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.SECOND)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        for cf_beat in range(len(cf)):
            strong_idx = cf_beat * 2
            if strong_idx < len(cp):
                intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
                assert consonant_interval_class(intv)

    @given(cf=cantus_firmus)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_doubled_length(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.SECOND)
        result = gen.generate()
        assume(result.feasible)
        assert len(result.voices[1]) == len(cf) * 2


class TestPropertySpecies3:
    """Property: species 3 strong beats are consonant, note count is 4x."""

    @given(cf=st.lists(midi_pitch, min_size=3, max_size=8))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_strong_beats_consonant(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.THIRD)
        result = gen.generate()
        assume(result.feasible)
        cp = result.voices[1]
        for cf_beat in range(len(cf)):
            strong_idx = cf_beat * 4
            if strong_idx < len(cp):
                intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
                assert consonant_interval_class(intv)

    @given(cf=st.lists(midi_pitch, min_size=3, max_size=8))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_quadrupled_length(self, cf: list) -> None:
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.THIRD)
        result = gen.generate()
        assume(result.feasible)
        assert len(result.voices[1]) == len(cf) * 4


class TestPropertyRules:
    """Property tests for individual rule functions."""

    @given(
        a=st.integers(0, 127),
        b=st.integers(0, 127),
    )
    def test_consonant_interval_class_returns_bool(self, a: int, b: int) -> None:
        """consonant_interval_class always returns a bool."""
        intv = abs(a - b) % 12
        result = consonant_interval_class(intv)
        assert isinstance(result, bool)

    @given(
        voice_a=st.lists(st.integers(0, 127), min_size=2, max_size=20),
        voice_b=st.lists(st.integers(0, 127), min_size=2, max_size=20),
    )
    def test_parallel_fifths_never_crashes(self, voice_a: list, voice_b: list) -> None:
        """no_parallel_fifths should never raise an exception."""
        n = min(len(voice_a), len(voice_b))
        assume(n >= 2)
        beats = list(range(n))
        result = no_parallel_fifths(voice_a[:n], voice_b[:n], beats)
        assert result in (SAT, UNSAT)

    @given(
        voice_a=st.lists(st.integers(0, 127), min_size=2, max_size=20),
        voice_b=st.lists(st.integers(0, 127), min_size=2, max_size=20),
    )
    def test_parallel_octaves_never_crashes(self, voice_a: list, voice_b: list) -> None:
        """no_parallel_octaves should never raise an exception."""
        n = min(len(voice_a), len(voice_b))
        assume(n >= 2)
        beats = list(range(n))
        result = no_parallel_octaves(voice_a[:n], voice_b[:n], beats)
        assert result in (SAT, UNSAT)
