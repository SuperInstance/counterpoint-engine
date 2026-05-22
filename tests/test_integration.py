"""Integration tests: full pipeline from generation to tensor-midi output."""

import pytest

from counterpoint_engine.generator import CounterpointGenerator, Species
from counterpoint_engine.laman_counterpoint import CounterpointGraph
from counterpoint_engine.tensor_output import (
    voices_to_tensor_events,
    voice_leading_to_sidechannels,
)
from counterpoint_engine.rules import (
    SAT,
    no_parallel_fifths,
    no_parallel_octaves,
    proper_resolution,
    max_leap_seventh,
    consonant_interval,
    voice_independence,
)


# A Bach-style cantus firmus in C major (8 beats)
BACH_CANTUS = [60, 64, 65, 64, 62, 67, 65, 64]  # C E F E D G F E


class TestFullPipeline:
    def test_generate_2_voice_and_output(self):
        """Generate 2-voice counterpoint and convert to tensor-midi."""
        gen = CounterpointGenerator(cantus_firmus=BACH_CANTUS)
        cp = gen.generate()
        assert cp is not None

        voices = [BACH_CANTUS, cp]
        tensor_events, midi_events = voices_to_tensor_events(voices)
        assert len(tensor_events) == len(midi_events) == len(BACH_CANTUS) * 2

    def test_generate_4_voice_fugue_excerpt(self):
        """Generate a 4-voice fugue excerpt and verify all properties."""
        gen = CounterpointGenerator(cantus_firmus=BACH_CANTUS)
        voices = gen.generate_n_voices(4)
        assert voices is not None
        assert len(voices) == 4
        assert all(len(v) == len(BACH_CANTUS) for v in voices)

        # 1. No parallel fifths between any pair
        beats = list(range(len(BACH_CANTUS)))
        for i in range(4):
            for j in range(i + 1, 4):
                assert no_parallel_fifths(voices[i], voices[j], beats) == SAT

        # 2. No parallel octaves between any pair
        for i in range(4):
            for j in range(i + 1, 4):
                assert no_parallel_octaves(voices[i], voices[j], beats) == SAT

        # 3. All intervals consonant at every beat
        for b in beats:
            for i in range(4):
                for j in range(i + 1, 4):
                    assert consonant_interval(voices[i], voices[j], b) == SAT

        # 4. Proper resolution in every voice
        for v in voices:
            for b in beats:
                assert proper_resolution(v, b) == SAT

        # 5. Max leap respected in every voice
        for v in voices:
            for b in beats:
                assert max_leap_seventh(v, b) == SAT

        # 6. Laman rigidity of 4-voice constraint graph
        g = CounterpointGraph(4)
        assert g.is_minimally_rigid()
        assert voice_independence(g.verify_rigidity()) == SAT

        # 7. Convert to tensor-midi
        tensor_events, midi_events = voices_to_tensor_events(voices)
        assert len(tensor_events) == len(midi_events) == len(BACH_CANTUS) * 4

        # 8. Side-channel analysis
        for b in beats:
            side = voice_leading_to_sidechannels(voices, b)
            # Every pair should have a side-channel assignment
            assert len(side) == 6  # C(4,2) = 6

    def test_laman_rigidity_scaling(self):
        """Verify Laman rigidity holds for 2 through 8 voices."""
        for n in range(2, 9):
            gen = CounterpointGenerator(cantus_firmus=BACH_CANTUS)
            voices = gen.generate_n_voices(n)
            if voices is None:
                continue  # Some N may be unsatisfiable with strict rules
            g = CounterpointGraph(n)
            assert g.edge_count() == 2 * n - 3
            assert g.verify_rigidity()

    def test_tensor_event_byte_size(self):
        """TensorMIDIEvent must be exactly 4 bytes."""
        from counterpoint_engine.tensor_output import TensorMIDIEvent
        e = TensorMIDIEvent(10, 20, 5, 128)
        assert len(e.to_bytes()) == 4

    def test_theorem_counterpoint_equals_laman(self):
        """Core thesis: counterpoint rigidity ↔ Laman rigidity.

        For N voices, the constraint graph must have exactly 2N-3 edges
        and satisfy the Laman subset condition.
        """
        for n in [3, 4, 5, 6]:
            g = CounterpointGraph(n)
            assert g.edge_count() == 2 * n - 3
            assert g.verify_rigidity()
            # Every edge must correspond to at least one constraint
            for edge in g.edges:
                assert edge in g.constraints
                assert len(g.constraints[edge]) >= 1
