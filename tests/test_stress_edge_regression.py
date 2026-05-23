"""Stress tests: generate many counterpoints in a loop, edge cases, benchmarks."""

import time
import pytest

from counterpoint_engine.generator import (
    CounterpointGenerator,
    CounterpointResult,
    Species,
    Scale,
    VoiceRange,
)
from counterpoint_engine.rules import (
    SAT,
    consonant_interval,
    consonant_interval_class,
    no_parallel_fifths,
    no_parallel_octaves,
    max_leap_seventh,
    proper_resolution,
)
from counterpoint_engine.exceptions import (
    RangeViolationError,
    InvalidInputError,
)


C_MAJOR_CANTUS = [60, 62, 64, 65, 67, 65, 64, 62]


# ---------------------------------------------------------------------------
# Stress tests: generate 100 counterpoints
# ---------------------------------------------------------------------------

class TestStressSpecies1:
    """Generate 100 species-1 counterpoints and verify none crash."""

    def test_100_generations_species1(self) -> None:
        for i in range(100):
            # Vary the cantus firmus slightly each time
            cf = [60 + (i % 5), 62, 64, 65, 67, 65, 64, 62]
            gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
            result = gen.generate()
            assert result.feasible, f"Iteration {i}: generation failed for cf={cf}"
            assert len(result.voices[1]) == len(cf)

    def test_100_generations_different_scales(self) -> None:
        """Generate with different tonic pitch classes."""
        for tonic in range(12):
            scale = Scale(tonic=tonic, mode="major")
            cf = [60 + tonic, 62 + tonic, 64 + tonic, 65 + tonic, 67 + tonic]
            # Only use pitches in MIDI range
            cf = [p for p in cf if 0 <= p <= 127]
            if len(cf) < 3:
                continue
            gen = CounterpointGenerator(
                cantus_firmus=cf, species=Species.FIRST, scale=scale,
            )
            result = gen.generate()
            assert result.feasible, f"Tonic={tonic} failed"


class TestStressSpecies2:
    def test_50_generations_species2(self) -> None:
        for i in range(50):
            cf = [60 + (i % 3), 62, 64, 65, 67, 65, 64, 62]
            gen = CounterpointGenerator(cantus_firmus=cf, species=Species.SECOND)
            result = gen.generate()
            assert result.feasible, f"Iteration {i}: species 2 failed"
            assert len(result.voices[1]) == len(cf) * 2


class TestStressSpecies3:
    def test_25_generations_species3(self) -> None:
        for i in range(25):
            cf = [60 + (i % 3), 62, 64, 65, 67, 65, 64, 62]
            gen = CounterpointGenerator(cantus_firmus=cf, species=Species.THIRD)
            result = gen.generate()
            assert result.feasible, f"Iteration {i}: species 3 failed"
            assert len(result.voices[1]) == len(cf) * 4


class TestStressSpecies4:
    def test_25_generations_species4(self) -> None:
        for i in range(25):
            cf = [60 + (i % 3), 62, 64, 65, 67, 65, 64, 62]
            gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FOURTH)
            result = gen.generate()
            assert result.feasible, f"Iteration {i}: species 4 failed"
            assert len(result.voices[1]) == len(cf)


class TestStressSpecies5:
    def test_25_generations_species5(self) -> None:
        for i in range(25):
            cf = [60 + (i % 3), 62, 64, 65, 67, 65, 64, 62]
            gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIFTH)
            result = gen.generate()
            assert result.feasible, f"Iteration {i}: species 5 failed"
            assert len(result.voices[1]) >= len(cf)


# ---------------------------------------------------------------------------
# Edge cases: extreme cantus firmus lengths
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: very short, very long, single voice, many voices."""

    def test_cantus_length_2(self) -> None:
        gen = CounterpointGenerator(cantus_firmus=[60, 67], species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == 2

    def test_cantus_length_2_species2(self) -> None:
        gen = CounterpointGenerator(cantus_firmus=[60, 67], species=Species.SECOND)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == 4

    def test_cantus_length_50(self) -> None:
        """Long cantus firmus: generate a 50-beat species 1 counterpoint."""
        cf = [60 + (i % 7) * 2 for i in range(50)]
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == 50
        # Verify all constraints
        cp = result.voices[1]
        beats = list(range(50))
        assert no_parallel_fifths(cf, cp, beats) == SAT
        assert no_parallel_octaves(cf, cp, beats) == SAT

    def test_single_voice(self) -> None:
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(1)
        assert result.feasible
        assert result.n_voices == 1
        assert len(result.voices) == 1

    def test_8_voices(self) -> None:
        """Generate 8-voice counterpoint."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(8)
        assert result.feasible
        assert len(result.voices) == 8
        # Check all voice pairs for parallel intervals
        beats = list(range(len(C_MAJOR_CANTUS)))
        for i in range(8):
            for j in range(i + 1, 8):
                assert no_parallel_fifths(result.voices[i], result.voices[j], beats) == SAT
                assert no_parallel_octaves(result.voices[i], result.voices[j], beats) == SAT

    def test_narrow_voice_range(self) -> None:
        """Very narrow range (only a 5th)."""
        vr = VoiceRange(min_pitch=60, max_pitch=67)
        gen = CounterpointGenerator(
            cantus_firmus=[60, 62, 64, 65],
            species=Species.FIRST,
            voice_range=vr,
        )
        result = gen.generate()
        assert result.feasible
        for p in result.voices[1]:
            assert 60 <= p <= 67

    def test_empty_cantus_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CounterpointGenerator(cantus_firmus=[], species=Species.FIRST)

    def test_invalid_midi_pitch_rejected(self) -> None:
        with pytest.raises(RangeViolationError):
            CounterpointGenerator(cantus_firmus=[60, -1, 64])

    def test_non_int_pitch_rejected(self) -> None:
        with pytest.raises(TypeError):
            CounterpointGenerator(cantus_firmus=[60, 62.5, 64])

    def test_invalid_species_rejected(self) -> None:
        with pytest.raises(ValueError, match="Species enum"):
            CounterpointGenerator(cantus_firmus=[60, 62], species="first")  # type: ignore

    def test_invalid_n_voices(self) -> None:
        gen = CounterpointGenerator(cantus_firmus=[60, 62])
        with pytest.raises(ValueError, match="n_voices must be >= 1"):
            gen.generate_n_voices(0)

    def test_minor_scale(self) -> None:
        scale = Scale(tonic=0, mode="minor")
        cf = [60, 62, 63, 65, 67, 68, 70, 67]  # C minor scale
        gen = CounterpointGenerator(cantus_firmus=cf, scale=scale)
        result = gen.generate()
        assert result.feasible

    def test_very_high_cantus(self) -> None:
        cf = [84, 86, 88, 89, 91]  # C6 range
        vr = VoiceRange(min_pitch=72, max_pitch=96)
        gen = CounterpointGenerator(cantus_firmus=cf, voice_range=vr)
        result = gen.generate()
        assert result.feasible

    def test_very_low_cantus(self) -> None:
        cf = [36, 38, 40, 41, 43]  # C2 range
        vr = VoiceRange(min_pitch=24, max_pitch=55)
        gen = CounterpointGenerator(cantus_firmus=cf, voice_range=vr)
        result = gen.generate()
        assert result.feasible


# ---------------------------------------------------------------------------
# Regression tests: pin the 5 previously-failing multi-voice tests
# ---------------------------------------------------------------------------

class TestMultiVoiceRegression:
    """Pin the multi-voice generation tests so they can't regress.

    These were the 5 previously-failing multi-voice tests that are now fixed.
    Each test verifies a specific property of multi-voice generation.
    """

    def test_3_voice_all_consonant(self) -> None:
        """REGRESSION: 3-voice counterpoint must have consonant intervals
        between all voice pairs at every beat."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(3)
        assert result.feasible
        for b in range(len(C_MAJOR_CANTUS)):
            for i in range(3):
                for j in range(i + 1, 3):
                    assert consonant_interval(
                        result.voices[i], result.voices[j], b
                    ) == SAT, (
                        f"Dissonance between voice {i} and {j} at beat {b}"
                    )

    def test_4_voice_no_parallel_fifths(self) -> None:
        """REGRESSION: 4-voice counterpoint must have no parallel fifths
        between any pair."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(4)
        assert result.feasible
        beats = list(range(len(C_MAJOR_CANTUS)))
        for i in range(4):
            for j in range(i + 1, 4):
                assert no_parallel_fifths(
                    result.voices[i], result.voices[j], beats
                ) == SAT, (
                    f"Parallel fifths between voice {i} and {j}"
                )

    def test_4_voice_no_parallel_octaves(self) -> None:
        """REGRESSION: 4-voice counterpoint must have no parallel octaves
        between any pair."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(4)
        assert result.feasible
        beats = list(range(len(C_MAJOR_CANTUS)))
        for i in range(4):
            for j in range(i + 1, 4):
                assert no_parallel_octaves(
                    result.voices[i], result.voices[j], beats
                ) == SAT, (
                    f"Parallel octaves between voice {i} and {j}"
                )

    def test_5_voice_max_leap_all_voices(self) -> None:
        """REGRESSION: every voice in a 5-voice texture respects max leap."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(5)
        assert result.feasible
        for voice in result.voices:
            for b in range(len(voice)):
                assert max_leap_seventh(voice, b) == SAT

    def test_5_voice_proper_resolution_all_voices(self) -> None:
        """REGRESSION: every voice in a 5-voice texture resolves leading tones."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(5)
        assert result.feasible
        for voice in result.voices:
            for b in range(len(voice)):
                assert proper_resolution(voice, b) == SAT


# ---------------------------------------------------------------------------
# Benchmarks: timing for common operations
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Timing benchmarks for common operations."""

    def test_species1_generation_time(self) -> None:
        """Species 1 generation for an 8-beat CF should be < 1 second."""
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIRST)
        result = gen.generate()
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 1.0, f"Species 1 took {elapsed:.3f}s"

    def test_species2_generation_time(self) -> None:
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.SECOND)
        result = gen.generate()
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 2.0, f"Species 2 took {elapsed:.3f}s"

    def test_species3_generation_time(self) -> None:
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.THIRD)
        result = gen.generate()
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 5.0, f"Species 3 took {elapsed:.3f}s"

    def test_species4_generation_time(self) -> None:
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 10.0, f"Species 4 took {elapsed:.3f}s"

    def test_3_voice_generation_time(self) -> None:
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(3)
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 5.0, f"3-voice took {elapsed:.3f}s"

    def test_4_voice_generation_time(self) -> None:
        start = time.perf_counter()
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(4)
        elapsed = time.perf_counter() - start
        assert result.feasible
        assert elapsed < 15.0, f"4-voice took {elapsed:.3f}s"

    def test_rule_checking_time(self) -> None:
        """Checking all rules on a 50-beat counterpoint should be fast."""
        cf = [60 + (i % 7) * 2 for i in range(50)]
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        beats = list(range(50))

        start = time.perf_counter()
        for _ in range(100):
            no_parallel_fifths(cf, cp, beats)
            no_parallel_octaves(cf, cp, beats)
            for b in beats:
                consonant_interval(cf, cp, b)
                max_leap_seventh(cp, b)
                proper_resolution(cp, b)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"100x rule check took {elapsed:.3f}s"
