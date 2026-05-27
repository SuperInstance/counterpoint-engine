"""Tests for counterpoint_engine.canon — Canon generation."""

from counterpoint_engine.canon import (
    CanonVoice,
    CanonGenerator,
    CanonResult,
    REST,
    make_follower,
    round_canon,
)
from counterpoint_engine.generator import Scale, VoiceRange


# Simple leader melodies
C_MAJOR_ASC = [60, 62, 64, 65, 67, 65, 64, 62]
SIMPLE_MELODY = [60, 62, 64, 65]


class TestCanonVoice:
    def test_defaults(self):
        cv = CanonVoice()
        assert cv.interval == 4
        assert cv.offset == 2
        assert cv.label == "follower"

    def test_custom(self):
        vr = VoiceRange(min_pitch=40, max_pitch=70)
        cv = CanonVoice(interval=7, offset=4, voice_range=vr, label="comes")
        assert cv.interval == 7
        assert cv.offset == 4
        assert cv.label == "comes"

    def test_negative_offset_raises(self):
        import pytest
        with pytest.raises(ValueError, match="offset must be >= 0"):
            CanonVoice(offset=-1)

    def test_zero_offset(self):
        cv = CanonVoice(offset=0)
        assert cv.offset == 0

    def test_frozen(self):
        cv = CanonVoice()
        import pytest
        with pytest.raises(AttributeError):
            cv.interval = 5  # type: ignore[misc]


class TestCanonGenerator:
    def test_basic_canon(self):
        cg = CanonGenerator(
            leader=C_MAJOR_ASC,
            followers=[CanonVoice(interval=4, offset=2)],
        )
        result = cg.generate()
        assert isinstance(result, CanonResult)
        assert len(result.voices) == 2
        # Follower starts 2 beats later, so total = 8 + 2 = 10
        assert result.total_beats == 10

    def test_leader_padded(self):
        cg = CanonGenerator(
            leader=[60, 62, 64],
            followers=[CanonVoice(interval=0, offset=1)],
        )
        result = cg.generate()
        # Leader should be padded: [60, 62, 64, REST]
        assert result.leader == [60, 62, 64, REST]

    def test_follower_transposition(self):
        cg = CanonGenerator(
            leader=[60, 62, 64],
            followers=[CanonVoice(interval=7, offset=1)],
        )
        result = cg.generate()
        follower = result.voices[1]
        # Offset 1: [REST, 67, 69, 71]
        assert follower[0] == REST
        assert follower[1] == 67
        assert follower[2] == 69
        assert follower[3] == 71

    def test_follower_at_unison(self):
        cg = CanonGenerator(
            leader=[60, 62, 64],
            followers=[CanonVoice(interval=0, offset=2)],
        )
        result = cg.generate()
        follower = result.voices[1]
        # Offset 2: [REST, REST, 60, 62, 64]
        assert follower[0] == REST
        assert follower[1] == REST
        assert follower[2] == 60
        assert follower[3] == 62
        assert follower[4] == 64

    def test_multi_follower(self):
        cg = CanonGenerator(
            leader=SIMPLE_MELODY,
            followers=[
                CanonVoice(interval=4, offset=2),
                CanonVoice(interval=7, offset=4),
            ],
        )
        result = cg.generate()
        assert len(result.voices) == 3
        assert result.total_beats == 8  # 4 + max(2, 4)

    def test_empty_leader_raises(self):
        import pytest
        with pytest.raises(ValueError, match="leader melody must not be empty"):
            CanonGenerator(leader=[])

    def test_non_int_leader_raises(self):
        import pytest
        with pytest.raises(TypeError, match="must be int"):
            CanonGenerator(leader=[60, "bad", 64]).generate()

    def test_no_followers(self):
        cg = CanonGenerator(leader=SIMPLE_MELODY, followers=[])
        result = cg.generate()
        assert len(result.voices) == 1  # Just leader
        assert result.total_beats == 4

    def test_validation_passes_for_good_canon(self):
        cg = CanonGenerator(
            leader=C_MAJOR_ASC,
            followers=[CanonVoice(interval=4, offset=2)],
        )
        result = cg.generate()
        # Should have a score
        assert 0.0 <= result.constraint_score <= 1.0

    def test_skip_validation(self):
        cg = CanonGenerator(
            leader=C_MAJOR_ASC,
            followers=[CanonVoice(interval=4, offset=2)],
            validate=False,
        )
        result = cg.generate()
        assert result.constraint_score == 1.0

    def test_repr(self):
        cg = CanonGenerator(leader=SIMPLE_MELODY)
        assert "CanonGenerator" in repr(cg)


class TestMakeFollower:
    def test_basic(self):
        result = make_follower(SIMPLE_MELODY, interval=7, offset=1)
        assert len(result.voices) == 2
        assert result.voices[1][0] == REST
        assert result.voices[1][1] == 67

    def test_custom_range(self):
        vr = VoiceRange(min_pitch=50, max_pitch=80)
        result = make_follower(SIMPLE_MELODY, voice_range=vr)
        assert isinstance(result, CanonResult)


class TestRoundCanon:
    def test_4_voice_round(self):
        result = round_canon(SIMPLE_MELODY, n_voices=4, offset=2)
        assert len(result.voices) == 4
        # Each voice offset by 2, so total = 4 + 2*3 = 10
        assert result.total_beats == 10

    def test_2_voice_round(self):
        result = round_canon([60, 62, 64], n_voices=2, offset=1)
        assert len(result.voices) == 2
        assert result.total_beats == 4

    def test_follower_pitches_match_leader(self):
        result = round_canon([60, 62, 64, 65], n_voices=3, offset=2)
        # Voice 2 should enter at offset 4 with same pitches
        v2 = result.voices[2]
        assert v2[4] == 60
        assert v2[5] == 62


class TestCanonValidation:
    def test_parallel_fifths_detected(self):
        """Canon validation checks overlapping beats for consonance."""
        # Use a melody that creates dissonant overlap with the follower
        leader = [60, 61]  # C-C#
        cg = CanonGenerator(
            leader=leader,
            followers=[CanonVoice(interval=0, offset=0)],
            validate=True,
        )
        result = cg.generate()
        # Same melody at same offset = parallel octaves
        assert not result.feasible or result.constraint_score < 1.0

    def test_good_canon_is_feasible(self):
        """A well-spaced canon should pass basic validation."""
        cg = CanonGenerator(
            leader=[60, 62, 64, 65, 67, 69, 67, 65],
            followers=[CanonVoice(interval=4, offset=3)],
        )
        result = cg.generate()
        # With 3-beat offset and 4th transposition, overlapping beats
        # should generally be consonant
        assert result.constraint_score > 0.0


class TestCanonWithCounterpoint:
    def test_generate_with_counterpoint(self):
        cg = CanonGenerator(
            leader=C_MAJOR_ASC,
            followers=[CanonVoice(interval=4, offset=2)],
        )
        result = cg.generate_with_counterpoint()
        assert result.feasible
        assert len(result.voices) == 2
