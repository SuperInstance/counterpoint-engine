"""Tests for counterpoint_engine.generator — constrained generation."""


from counterpoint_engine.generator import (
    CounterpointGenerator,
    CounterpointResult,
    Species,
    Scale,
    VoiceRange,
)
from counterpoint_engine.rules import (
    SAT,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
)


# A simple C-major cantus firmus (ascending scale, very constrained)
C_MAJOR_CANTUS = [60, 62, 64, 65, 67, 65, 64, 62]  # C D E F G F E D


class TestCounterpointGenerator:
    def test_basic_generation(self):
        gen = CounterpointGenerator(
            cantus_firmus=C_MAJOR_CANTUS,
            species=Species.FIRST,
        )
        result = gen.generate()
        assert isinstance(result, CounterpointResult)
        assert result.feasible
        assert len(result.voices) == 2
        assert len(result.voices[1]) == len(C_MAJOR_CANTUS)

    def test_generated_satisfies_parallel_fifths(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        beats = list(range(len(cp)))
        assert no_parallel_fifths(C_MAJOR_CANTUS, cp, beats) == SAT

    def test_generated_satisfies_parallel_octaves(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        beats = list(range(len(cp)))
        assert no_parallel_octaves(C_MAJOR_CANTUS, cp, beats) == SAT

    def test_generated_satisfies_consonance(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        for b in range(len(cp)):
            assert consonant_interval(C_MAJOR_CANTUS, cp, b) == SAT

    def test_generated_satisfies_max_leap(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        for b in range(len(cp)):
            assert max_leap_seventh(cp, b) == SAT

    def test_generated_satisfies_proper_resolution(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        for b in range(len(cp)):
            assert proper_resolution(cp, b) == SAT

    def test_custom_scale(self):
        scale = Scale(tonic=2, mode="major")  # D major
        gen = CounterpointGenerator(
            cantus_firmus=[62, 64, 66, 67, 69, 67, 66, 64],
            scale=scale,
        )
        result = gen.generate()
        assert result.feasible

    def test_voice_range_restricts(self):
        vr = VoiceRange(min_pitch=55, max_pitch=70)
        gen = CounterpointGenerator(
            cantus_firmus=C_MAJOR_CANTUS,
            voice_range=vr,
        )
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        assert all(vr.min_pitch <= p <= vr.max_pitch for p in cp)

    def test_short_cantus(self):
        gen = CounterpointGenerator(cantus_firmus=[60, 64, 67])
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == 3

    def test_all_constraints_combined(self):
        """Bach-style combined constraint check."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        beats = list(range(len(cp)))
        assert no_parallel_fifths(C_MAJOR_CANTUS, cp, beats) == SAT
        assert no_parallel_octaves(C_MAJOR_CANTUS, cp, beats) == SAT
        for b in beats:
            assert consonant_interval(C_MAJOR_CANTUS, cp, b) == SAT
            assert max_leap_seventh(cp, b) == SAT
            assert proper_resolution(cp, b) == SAT

    def test_result_has_constraint_counts(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate()
        assert result.feasible
        assert result.constraints_total > 0
        assert result.constraints_satisfied > 0
        assert result.species == 1
        assert result.n_voices == 2


class TestMultiVoiceGeneration:
    def test_three_voices(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(3)
        assert isinstance(result, CounterpointResult)
        assert result.feasible
        assert len(result.voices) == 3
        assert all(len(v) == len(C_MAJOR_CANTUS) for v in result.voices)

    def test_four_voices(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
        result = gen.generate_n_voices(4)
        assert result.feasible
        assert len(result.voices) == 4

    def test_laman_rigidity_of_generated(self):
        """The constraint graph for N-voice counterpoint must be Laman."""
        from counterpoint_engine.laman_counterpoint import CounterpointGraph
        for n in [3, 4, 5]:
            gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS)
            result = gen.generate_n_voices(n)
            assert result.feasible
            g = CounterpointGraph(n)
            assert g.is_minimally_rigid()
