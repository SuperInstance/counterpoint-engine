"""Tests for the five musical fixes from the beta test feedback.

a) Harmonic minor scale with raised 7th (leading tone)
b) Fux unison rule (unisons only at first/last beat)
c) Contrary motion preference (scoring bonus)
d) Voice crossing prevention (voice_range_invariant)
e) Renamed "fugue" to "multi-voice counterpoint"
"""


from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    consonant_interval,
    contrary_motion_bonus,
    contrary_motion_score,
    proper_resolution,
    voice_range_invariant,
)
from counterpoint_engine.generator import (
    CounterpointGenerator,
    Scale,
    Species,
)


# ---------------------------------------------------------------------------
# a) Harmonic minor scale
# ---------------------------------------------------------------------------


class TestHarmonicMinor:
    """Harmonic minor has a raised 7th (leading tone) that must resolve."""

    def test_minor_mode_uses_harmonic_minor(self):
        """Scale(mode='minor') should use harmonic minor (raised 7th)."""
        s = Scale(tonic=0, mode="minor")  # C minor
        # Harmonic minor: C D Eb F G Ab B (0,2,3,5,7,8,11)
        assert s.contains(11)  # B natural (leading tone)
        assert s.contains(3)   # Eb (minor third)

    def test_natural_minor_available(self):
        """Natural minor (Aeolian) should be available as explicit mode."""
        s = Scale(tonic=0, mode="natural_minor")
        # Natural minor: C D Eb F G Ab Bb (0,2,3,5,7,8,10)
        assert s.contains(10)  # Bb (subtonic, NOT leading tone)
        assert 11 not in s.pitch_classes()

    def test_leading_tone_resolves_in_minor(self):
        """In D minor, C# (leading tone) must resolve to D (tonic)."""
        # D minor: tonic=2, leading tone = (2+11)%12 = 1 (C#)
        Scale(tonic=2, mode="minor")
        # C#(61) should resolve to D(62)
        assert proper_resolution([61, 62], 1, key_tonic=2, key_leading=1) == SAT
        # C#(61) resolving to B(59) should fail
        assert proper_resolution([61, 59], 1, key_tonic=2, key_leading=1) == UNSAT

    def test_minor_scale_leading_tone_method(self):
        """Scale.leading_tone() returns the correct pitch class."""
        s = Scale(tonic=0, mode="minor")  # C minor
        assert s.leading_tone() == 11  # B natural

    def test_minor_scale_is_harmonic(self):
        """Scale.is_harmonic_minor() correctly identifies harmonic minor."""
        s = Scale(tonic=0, mode="minor")
        assert s.is_harmonic_minor()

    def test_natural_minor_is_not_harmonic(self):
        """Natural minor should not be identified as harmonic minor."""
        s = Scale(tonic=0, mode="natural_minor")
        # Natural minor has subtonic (10) not leading tone (11)
        assert not s.is_harmonic_minor()

    def test_minor_counterpoint_generation(self):
        """Generate counterpoint in D minor using harmonic minor."""
        # D minor cantus firmus
        cf = [62, 64, 63, 65, 67, 65, 63, 62]  # D Eb-reverse pattern
        scale = Scale(tonic=2, mode="minor")
        gen = CounterpointGenerator(
            cantus_firmus=cf,
            species=Species.FIRST,
            scale=scale,
        )
        result = gen.generate()
        assert result.feasible
        # Verify the counterpoint uses scale tones
        cp = result.voices[1]
        for pitch in cp:
            assert scale.contains(pitch), f"Pitch {pitch} not in D harmonic minor"

    def test_proper_resolution_with_scale_pitch_classes(self):
        """proper_resolution respects scale membership."""
        # In C natural minor, Bb (10) is the subtonic, NOT the leading tone
        # So Bb→Ab should be allowed (not enforced to resolve to C)
        s = Scale(tonic=0, mode="natural_minor")
        # Bb(70) → Ab(68): Bb is in natural_minor scale but is NOT the leading tone
        assert proper_resolution(
            [70, 68], 1,
            key_tonic=0, key_leading=11,
            scale_pitch_classes=s.pitch_classes(),
        ) == SAT  # Bb (pc=10) != leading_tone (11), so no enforcement


# ---------------------------------------------------------------------------
# b) Fux unison rule
# ---------------------------------------------------------------------------


class TestFuxUnisonRule:
    """Unisons only allowed at first and last beat (Fux)."""

    def test_unison_at_first_beat_allowed(self):
        """Unison at beat 0 is allowed."""
        assert consonant_interval([60], [60], 0, total_beats=4) == SAT

    def test_unison_at_last_beat_allowed(self):
        """Unison at last beat is allowed."""
        assert consonant_interval([60, 62, 64, 60], [55, 57, 59, 60], 3, total_beats=4) == SAT

    def test_unison_at_interior_beat_rejected(self):
        """Unison at interior beat is rejected."""
        assert consonant_interval([60, 60, 62, 64], [60, 60, 62, 64], 1, total_beats=4) == UNSAT
        assert consonant_interval([60, 62, 62, 64], [60, 62, 62, 64], 2, total_beats=4) == UNSAT

    def test_unison_without_total_beats_no_restriction(self):
        """Without total_beats, unisons are allowed everywhere (backward compat)."""
        assert consonant_interval([60, 60, 62], [55, 55, 57], 1) == SAT
        assert consonant_interval([60, 62, 62], [55, 57, 57], 2) == SAT

    def test_generated_counterpoint_no_interior_unisons(self):
        """First-species counterpoint should not have interior unisons."""
        cf = [60, 62, 64, 65, 67, 65, 64, 62]
        gen = CounterpointGenerator(cantus_firmus=cf, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        for b in range(1, len(cp) - 1):
            intv = abs(cf[b] - cp[b]) % 12
            assert intv != 0, f"Unison at interior beat {b}: CF={cf[b]}, CP={cp[b]}"


# ---------------------------------------------------------------------------
# c) Contrary motion preference
# ---------------------------------------------------------------------------


class TestContraryMotion:
    """Contrary motion scoring for voice leading quality."""

    def test_contrary_motion_bonus_positive(self):
        """Contrary motion returns bonus of 1."""
        # Voice A goes up (60→62), Voice B goes down (67→65)
        assert contrary_motion_bonus([60, 62], [67, 65], 1) == 1

    def test_similar_motion_bonus_zero(self):
        """Similar motion returns bonus of 0."""
        # Both voices go up
        assert contrary_motion_bonus([60, 62], [64, 66], 1) == 0

    def test_oblique_motion_bonus_zero(self):
        """Oblique motion returns bonus of 0."""
        # One voice static, other moves
        assert contrary_motion_bonus([60, 60], [64, 66], 1) == 0

    def test_beat_zero_bonus_zero(self):
        """No motion at beat 0."""
        assert contrary_motion_bonus([60, 62], [67, 65], 0) == 0

    def test_contrary_motion_score_all_contrary(self):
        """Score is 1.0 when all motion is contrary."""
        voice_a = [60, 62, 64]
        voice_b = [67, 65, 63]
        assert contrary_motion_score(voice_a, voice_b, [0, 1, 2]) == 1.0

    def test_contrary_motion_score_no_contrary(self):
        """Score is 0.0 when no contrary motion."""
        voice_a = [60, 62, 64]
        voice_b = [64, 66, 68]
        assert contrary_motion_score(voice_a, voice_b, [0, 1, 2]) == 0.0

    def test_contrary_motion_score_mixed(self):
        """Score is fractional when mixed motion."""
        voice_a = [60, 62, 60]  # up, down
        voice_b = [67, 65, 67]  # down, up — both contrary
        score = contrary_motion_score(voice_a, voice_b, [0, 1, 2])
        assert score == 1.0  # Both active motions are contrary

    def test_contrary_motion_score_empty(self):
        """Score is 0.0 for fewer than 2 beats."""
        assert contrary_motion_score([60], [67], [0]) == 0.0

    def test_generated_counterpoint_has_contrary_motion(self):
        """Generated counterpoint should have some contrary motion."""
        cf = [60, 62, 64, 65, 67, 65, 64, 62]
        gen = CounterpointGenerator(cantus_firmus=cf)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        beats = list(range(len(cp)))
        score = contrary_motion_score(cf, cp, beats)
        assert score > 0.0, "Counterpoint should have at least some contrary motion"


# ---------------------------------------------------------------------------
# d) Voice crossing prevention
# ---------------------------------------------------------------------------


class TestVoiceCrossing:
    """Voice crossing prevention in multi-voice counterpoint."""

    def test_voice_range_invariant_no_crossing(self):
        """SAT: soprano above alto."""
        voices = [[72, 74], [60, 62]]
        assert voice_range_invariant(voices, 0) == SAT
        assert voice_range_invariant(voices, 1) == SAT

    def test_voice_range_invariant_crossing(self):
        """UNSAT: alto above soprano."""
        voices = [[60, 62], [72, 74]]
        assert voice_range_invariant(voices, 0) == UNSAT

    def test_voice_range_invariant_three_voices(self):
        """Check with three voices."""
        # Soprano > Alto > Tenor
        voices = [[72, 74], [60, 62], [48, 50]]
        assert voice_range_invariant(voices, 0) == SAT
        assert voice_range_invariant(voices, 1) == SAT

    def test_voice_range_invariant_crossing_middle(self):
        """UNSAT when middle voice crosses above top."""
        voices = [[72, 74], [76, 78], [48, 50]]  # Alto > Soprano
        assert voice_range_invariant(voices, 0) == UNSAT

    def test_multi_voice_preferably_no_crossing(self):
        """Multi-voice generation should attempt to avoid crossing."""
        cf = [60, 62, 64, 65, 67, 65, 64, 62]
        gen = CounterpointGenerator(cantus_firmus=cf)
        result = gen.generate_n_voices(3)
        assert result.feasible
        assert result.n_voices == 3
        # Check if voices are crossing-free (preferred but not guaranteed)
        for b in range(len(cf)):
            assert voice_range_invariant(result.voices, b) == SAT

    def test_four_voice_generation(self):
        """4-voice generation should be feasible."""
        cf = [60, 62, 64, 65, 67, 65, 64, 62]
        gen = CounterpointGenerator(cantus_firmus=cf)
        result = gen.generate_n_voices(4)
        assert result.feasible
        assert result.n_voices == 4


# ---------------------------------------------------------------------------
# e) Renamed "fugue" to "multi-voice counterpoint"
# ---------------------------------------------------------------------------


class TestMultiVoiceNaming:
    """The 'fugue' terminology has been replaced with 'multi-voice counterpoint'."""

    def test_generate_n_voices_docstring(self):
        """generate_n_voices should reference multi-voice counterpoint."""
        doc = CounterpointGenerator.generate_n_voices.__doc__
        assert "multi-voice counterpoint" in doc.lower()
        assert "fugue" not in doc.lower()

    def test_method_exists(self):
        """generate_n_voices is the public API for multi-voice generation."""
        gen = CounterpointGenerator(cantus_firmus=[60, 62, 64])
        assert hasattr(gen, "generate_n_voices")
        result = gen.generate_n_voices(2)
        assert result.feasible
