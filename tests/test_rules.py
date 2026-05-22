"""Tests for counterpoint_engine.rules — individual FLUX constraints."""

import pytest

from counterpoint_engine.rules import (
    SAT,
    UNSAT,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
    voice_independence,
)


# ---------------------------------------------------------------------------
# no_parallel_fifths
# ---------------------------------------------------------------------------

class TestNoParallelFifths:
    def test_empty_beats(self):
        assert no_parallel_fifths([60, 64], [67, 71], []) == SAT

    def test_single_beat(self):
        assert no_parallel_fifths([60, 64], [67, 71], [0]) == SAT

    def test_no_parallel_fifths(self):
        # C-E then D-F (intervals M3, M3 — not fifths)
        voice_a = [60, 62]  # C, D
        voice_b = [64, 65]  # E, F
        assert no_parallel_fifths(voice_a, voice_b, [0, 1]) == SAT

    def test_parallel_fifths_detected(self):
        # C-G then D-A (both perfect fifths, similar motion)
        voice_a = [60, 62]  # C, D
        voice_b = [67, 69]  # G, A
        assert no_parallel_fifths(voice_a, voice_b, [0, 1]) == UNSAT

    def test_consecutive_fifths_oblique(self):
        # C-G then C-A (fifth then sixth, oblique motion)
        voice_a = [60, 60]
        voice_b = [67, 69]
        assert no_parallel_fifths(voice_a, voice_b, [0, 1]) == SAT

    def test_fifths_contrary_motion(self):
        # C-G then D-F# (fifth then tritone, contrary motion)
        # Actually let's do: C-G then B-G (fifth by contrary? No)
        # C-G (P5), B-G (m6) — not parallel
        voice_a = [60, 59]
        voice_b = [67, 67]
        assert no_parallel_fifths(voice_a, voice_b, [0, 1]) == SAT


# ---------------------------------------------------------------------------
# no_parallel_octaves
# ---------------------------------------------------------------------------

class TestNoParallelOctaves:
    def test_empty(self):
        assert no_parallel_octaves([60], [72], []) == SAT

    def test_no_parallel_octaves(self):
        voice_a = [60, 62]
        voice_b = [64, 65]
        assert no_parallel_octaves(voice_a, voice_b, [0, 1]) == SAT

    def test_parallel_octaves(self):
        # C4-C5 then D4-D5
        voice_a = [60, 62]
        voice_b = [72, 74]
        assert no_parallel_octaves(voice_a, voice_b, [0, 1]) == UNSAT

    def test_static_octaves(self):
        # C4-C5 held
        voice_a = [60, 60]
        voice_b = [72, 72]
        assert no_parallel_octaves(voice_a, voice_b, [0, 1]) == UNSAT


# ---------------------------------------------------------------------------
# proper_resolution
# ---------------------------------------------------------------------------

class TestProperResolution:
    def test_beat_zero(self):
        assert proper_resolution([60, 62], 0) == SAT

    def test_leading_tone_resolves(self):
        # B (71) → C (72) in C major
        voice = [71, 72]
        assert proper_resolution(voice, 1) == SAT

    def test_leading_tone_fails(self):
        # B (71) → A (69) — bad resolution
        voice = [71, 69]
        assert proper_resolution(voice, 1) == UNSAT

    def test_non_leading_tone(self):
        # D (62) → E (64)
        voice = [62, 64]
        assert proper_resolution(voice, 1) == SAT

    def test_different_key(self):
        # In G major: F# (66) should resolve to G (67)
        voice = [66, 67]
        assert proper_resolution(voice, 1, key_tonic=7, key_leading=6) == SAT
        voice_bad = [66, 64]
        assert proper_resolution(voice_bad, 1, key_tonic=7, key_leading=6) == UNSAT


# ---------------------------------------------------------------------------
# max_leap_seventh
# ---------------------------------------------------------------------------

class TestMaxLeapSeventh:
    def test_beat_zero(self):
        assert max_leap_seventh([60, 72], 0) == SAT

    def test_small_leap(self):
        voice = [60, 64]  # M3 = 4 semitones
        assert max_leap_seventh(voice, 1) == SAT

    def test_minor_seventh_exactly(self):
        voice = [60, 70]  # m7 = 10 semitones
        assert max_leap_seventh(voice, 1) == SAT

    def test_major_seventh(self):
        voice = [60, 71]  # M7 = 11 semitones
        assert max_leap_seventh(voice, 1) == UNSAT

    def test_octave_leap(self):
        # Octave leaps are traditionally allowed in some contexts
        # Our rule reduces compound intervals, so 12 mod 12 = 0, which is <= 10
        voice = [60, 72]
        assert max_leap_seventh(voice, 1) == SAT

    def test_ninth_leap(self):
        voice = [60, 74]  # M9 = 14 semitones, simple = 2
        # Simple leap is M2 = 2 <= 10, but absolute leap 14 > 10
        # Rule: if simple > max AND leap > max, then UNSAT (except octave)
        # simple=2 <= 10, so SAT
        assert max_leap_seventh(voice, 1) == SAT


# ---------------------------------------------------------------------------
# consonant_interval
# ---------------------------------------------------------------------------

class TestConsonantInterval:
    def test_unison(self):
        assert consonant_interval([60], [60], 0) == SAT

    def test_major_third(self):
        assert consonant_interval([60], [64], 0) == SAT

    def test_perfect_fifth(self):
        assert consonant_interval([60], [67], 0) == SAT

    def test_major_second(self):
        assert consonant_interval([60], [62], 0) == UNSAT

    def test_tritone(self):
        assert consonant_interval([60], [66], 0) == UNSAT

    def test_minor_seventh(self):
        assert consonant_interval([60], [70], 0) == UNSAT

    def test_custom_allowed(self):
        # Allow seconds if we're being liberal
        assert consonant_interval([60], [62], 0, allowed=(0, 2, 4, 7)) == SAT


# ---------------------------------------------------------------------------
# voice_independence (Laman wrapper)
# ---------------------------------------------------------------------------

class TestVoiceIndependence:
    def test_rigid(self):
        assert voice_independence(True) == SAT

    def test_not_rigid(self):
        assert voice_independence(False) == UNSAT
