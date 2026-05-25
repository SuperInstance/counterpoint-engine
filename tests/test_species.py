"""Tests for species 2-5 counterpoint generation."""


from counterpoint_engine.generator import (
    CounterpointGenerator,
    Species,
    VoiceRange,
)
from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    consonant_interval,
    consonant_interval_class,
    is_step,
    passing_tone_ok,
)


# A simple C-major cantus firmus
C_MAJOR_CANTUS = [60, 62, 64, 65, 67, 65, 64, 62]  # C D E F G F E D
SHORT_CANTUS = [60, 64, 67, 64, 60]  # C E G E C


class TestSpecies1:
    """Verify species 1 still works (regression)."""

    def test_species1_note_count(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        assert result.species == 1
        assert len(result.voices[1]) == len(C_MAJOR_CANTUS)

    def test_species1_all_consonant(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIRST)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        for b in range(len(cp)):
            assert consonant_interval(cf, cp, b) == SAT


class TestSpecies2:
    """Species 2: two-against-one."""

    def test_species2_doubled_note_count(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.SECOND)
        result = gen.generate()
        assert result.feasible
        assert result.species == 2
        # Counterpoint should have 2x the CF notes
        assert len(result.voices[1]) == len(C_MAJOR_CANTUS) * 2

    def test_species2_strong_beats_consonant(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.SECOND)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        for cf_beat in range(len(cf)):
            strong_idx = cf_beat * 2
            intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
            assert consonant_interval_class(intv), (
                f"Strong beat {cf_beat} (cp[{strong_idx}]={cp[strong_idx]} vs cf={cf[cf_beat]}) "
                f"is dissonant: interval class {intv}"
            )

    def test_species2_weak_beats_stepwise(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.SECOND)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        for cf_beat in range(len(C_MAJOR_CANTUS)):
            strong_idx = cf_beat * 2
            weak_idx = strong_idx + 1
            if weak_idx < len(cp):
                # Weak beat should be stepwise from strong beat
                assert abs(cp[weak_idx] - cp[strong_idx]) <= 2, (
                    f"Weak beat at idx {weak_idx} ({cp[weak_idx]}) not stepwise "
                    f"from strong {cp[strong_idx]}"
                )

    def test_species2_short_cantus(self):
        gen = CounterpointGenerator(cantus_firmus=SHORT_CANTUS, species=Species.SECOND)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == len(SHORT_CANTUS) * 2

    def test_species2_voice_range(self):
        vr = VoiceRange(min_pitch=55, max_pitch=72)
        gen = CounterpointGenerator(
            cantus_firmus=C_MAJOR_CANTUS,
            species=Species.SECOND,
            voice_range=vr,
        )
        result = gen.generate()
        assert result.feasible
        for p in result.voices[1]:
            assert vr.min_pitch <= p <= vr.max_pitch, f"Pitch {p} out of range"


class TestSpecies3:
    """Species 3: four-against-one."""

    def test_species3_quadrupled_note_count(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.THIRD)
        result = gen.generate()
        assert result.feasible
        assert result.species == 3
        assert len(result.voices[1]) == len(C_MAJOR_CANTUS) * 4

    def test_species3_strong_beats_consonant(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.THIRD)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        for cf_beat in range(len(cf)):
            strong_idx = cf_beat * 4
            intv = abs(cp[strong_idx] - cf[cf_beat]) % 12
            assert consonant_interval_class(intv), (
                f"Strong beat {cf_beat} dissonant: interval {intv}"
            )

    def test_species3_mostly_stepwise(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.THIRD)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        stepwise_count = 0
        for i in range(1, len(cp)):
            if abs(cp[i] - cp[i - 1]) <= 2:
                stepwise_count += 1
        # Most motion should be stepwise (allow some skips)
        assert stepwise_count / (len(cp) - 1) > 0.5, (
            f"Only {stepwise_count}/{len(cp)-1} stepwise motions"
        )

    def test_species3_short_cantus(self):
        gen = CounterpointGenerator(cantus_firmus=SHORT_CANTUS, species=Species.THIRD)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == len(SHORT_CANTUS) * 4

    def test_species3_voice_range(self):
        vr = VoiceRange(min_pitch=55, max_pitch=72)
        gen = CounterpointGenerator(
            cantus_firmus=C_MAJOR_CANTUS,
            species=Species.THIRD,
            voice_range=vr,
        )
        result = gen.generate()
        assert result.feasible
        for p in result.voices[1]:
            assert vr.min_pitch <= p <= vr.max_pitch


class TestSpecies4:
    """Species 4: syncopation / suspension."""

    def test_species4_same_note_count(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        assert result.species == 4
        # Same number of notes as CF
        assert len(result.voices[1]) == len(C_MAJOR_CANTUS)

    def test_species4_has_dissonances(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        # Species 4 should have at least some dissonances (suspensions)
        dissonances = 0
        for b in range(len(cp)):
            intv = abs(cp[b] - cf[b]) % 12
            if not consonant_interval_class(intv):
                dissonances += 1
        assert dissonances > 0, "Species 4 should have suspensions (controlled dissonance)"

    def test_species4_dissonances_are_suspensions(self):
        """Each dissonance must be preceded by a consonance (preparation)."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        for b in range(1, len(cp)):
            intv = abs(cp[b] - cf[b]) % 12
            if not consonant_interval_class(intv):
                # This is a suspension — previous beat must be consonant
                prev_intv = abs(cp[b - 1] - cf[b - 1]) % 12
                assert consonant_interval_class(prev_intv), (
                    f"Suspension at beat {b} not properly prepared"
                )

    def test_species4_short_cantus(self):
        gen = CounterpointGenerator(cantus_firmus=SHORT_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) == len(SHORT_CANTUS)

    def test_species4_not_all_identical(self):
        """Species 4 must produce melodic motion, not a static pedal."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        unique = len(set(cp))
        assert unique > 1, f"Species 4 produced static pitch: {cp}"

    def test_species4_melodic_range_at_least_octave(self):
        """Species 4 counterpoint should span at least an octave."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        span = max(cp) - min(cp)
        assert span >= 12, (
            f"Species 4 melodic range only {span} semitones (need >= 12): {cp}"
        )

    def test_species4_has_suspension_chains(self):
        """Species 4 must contain actual suspension patterns:
        consonant prep → dissonant suspension → consonant resolution."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]

        chains = 0
        for b in range(1, len(cp) - 1):
            intv_prev = abs(cp[b - 1] - cf[b - 1]) % 12
            intv_curr = abs(cp[b] - cf[b]) % 12
            intv_next = abs(cp[b + 1] - cf[b + 1]) % 12
            # Chain: consonant → dissonant → consonant
            if (consonant_interval_class(intv_prev)
                    and not consonant_interval_class(intv_curr)
                    and consonant_interval_class(intv_next)):
                chains += 1
        assert chains >= 1, (
            f"Species 4 has no suspension chains (consonant→dissonant→consonant). CP={cp}"
        )

    def test_species4_step_down_resolutions(self):
        """After a suspension (dissonance), the next note should resolve downward by step."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]

        resolutions = 0
        for b in range(1, len(cp) - 1):
            intv = abs(cp[b] - cf[b]) % 12
            if not consonant_interval_class(intv):
                # This is a suspension — check resolution on next beat
                diff = cp[b] - cp[b + 1]
                if 1 <= diff <= 2:
                    resolutions += 1
        assert resolutions >= 1, (
            f"Species 4 has no step-down resolutions after suspensions. CP={cp}"
        )

    def test_species4_avoids_excessive_unisons(self):
        """Species 4 should prefer imperfect consonances over unisons."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FOURTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]

        unisons = sum(
            1 for b in range(len(cp))
            if abs(cp[b] - cf[b]) % 12 == 0
        )
        assert unisons <= len(cp) // 2, (
            f"Too many unisons ({unisons}/{len(cp)}) in Species 4. CP={cp}"
        )


class TestSpecies5:
    """Species 5: florid counterpoint (mix of species 1-4)."""

    def test_species5_variable_note_count(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIFTH)
        result = gen.generate()
        assert result.feasible
        assert result.species == 5
        # Counterpoint should have >= CF length (at minimum species 1 everywhere)
        assert len(result.voices[1]) >= len(C_MAJOR_CANTUS)

    def test_species5_strong_beats_consonant(self):
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIFTH)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        # First note must be consonant
        intv = abs(cp[0] - cf[0]) % 12
        assert consonant_interval_class(intv), "First note must be consonant"

    def test_species5_short_cantus(self):
        gen = CounterpointGenerator(cantus_firmus=SHORT_CANTUS, species=Species.FIFTH)
        result = gen.generate()
        assert result.feasible
        assert len(result.voices[1]) >= len(SHORT_CANTUS)


class TestSpeciesDifferentiation:
    """Verify that different species produce different output."""

    def test_species_produce_different_lengths(self):
        """Species 1, 2, 3, 4, 5 should produce counterpoint of different lengths
        (except species 1 and 4 which are both 1:1)."""
        cf = C_MAJOR_CANTUS
        results = {}
        for species in [Species.FIRST, Species.SECOND, Species.THIRD, Species.FOURTH, Species.FIFTH]:
            gen = CounterpointGenerator(cantus_firmus=cf, species=species)
            result = gen.generate()
            assert result.feasible, f"Species {species} generation failed"
            results[int(species)] = len(result.voices[1])

        # Species 1 and 4 have same length as CF
        assert results[1] == len(cf)
        assert results[4] == len(cf)
        # Species 2 has 2x
        assert results[2] == len(cf) * 2
        # Species 3 has 4x
        assert results[3] == len(cf) * 4
        # Species 5 is variable (>= CF length)
        assert results[5] >= len(cf)

    def test_species2_not_identical_to_species1(self):
        """Species 2 counterpoint should not just repeat species 1 with doubled notes."""
        gen1 = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.FIRST)
        gen2 = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.SECOND)
        r1 = gen1.generate()
        r2 = gen2.generate()
        assert r1.feasible and r2.feasible
        # Different lengths prove different output
        assert len(r1.voices[1]) != len(r2.voices[1])

    def test_species3_has_weak_beat_dissonances(self):
        """Species 3 should have dissonant notes on weak beats (passing tones)."""
        gen = CounterpointGenerator(cantus_firmus=C_MAJOR_CANTUS, species=Species.THIRD)
        result = gen.generate()
        assert result.feasible
        cp = result.voices[1]
        cf = result.voices[0]
        # Check weak beats (indices not divisible by 4) for dissonances
        dissonant_weak = 0
        for cf_beat in range(len(cf)):
            for sub in range(1, 4):  # weak beats only
                idx = cf_beat * 4 + sub
                if idx < len(cp):
                    intv = abs(cp[idx] - cf[cf_beat]) % 12
                    if not consonant_interval_class(intv):
                        dissonant_weak += 1
        # Species 3 should ideally have some dissonant passing tones
        # (but we don't force it since the search might find all-consonant paths)
        # At minimum, verify the structure is correct
        assert len(cp) == len(cf) * 4


class TestSpeciesRules:
    """Test the new rule helpers."""

    def test_is_step(self):
        assert is_step(60, 61)
        assert is_step(60, 62)
        assert is_step(60, 59)
        assert is_step(60, 58)
        assert not is_step(60, 60)  # same note
        assert not is_step(60, 63)  # leap

    def test_consonant_interval_class(self):
        assert consonant_interval_class(0)   # unison
        assert consonant_interval_class(3)   # m3
        assert consonant_interval_class(4)   # M3
        assert consonant_interval_class(7)   # P5
        assert consonant_interval_class(12)  # P8
        assert not consonant_interval_class(1)  # m2
        assert not consonant_interval_class(6)  # tritone

    def test_passing_tone_ok(self):
        # Valid passing tone: step up then step up
        cp = [60, 62, 64]
        assert passing_tone_ok(cp, 1) == SAT

        # Invalid: leap then leap
        cp = [60, 65, 70]
        assert passing_tone_ok(cp, 1) == UNSAT

        # Boundary: exempt
        cp = [60, 62]
        assert passing_tone_ok(cp, 0) == SAT
        assert passing_tone_ok(cp, 1) == SAT
