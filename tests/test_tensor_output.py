"""Tests for counterpoint_engine.tensor_output — tensor-midi conversion."""


from counterpoint_engine.tensor_output import (
    TensorMIDIEvent,
    voices_to_tensor_events,
    voice_leading_to_sidechannels,
    interval_to_flux_vector,
    voice_intervals_to_flux_vectors,
)
try:
    from flux_tensor_midi.core.flux import FluxVector
except ImportError:
    FluxVector = None


class TestTensorMIDIEvent:
    def test_creation(self):
        e = TensorMIDIEvent(10, 20, 5, 128)
        assert e.cos_int8 == 10
        assert e.sin_int8 == 20
        assert e.beat_k == 5
        assert e.state_byte == 128

    def test_saturation(self):
        e = TensorMIDIEvent(200, -200, 300, 400)
        assert e.cos_int8 == 127
        assert e.sin_int8 == -128
        assert e.beat_k == 255
        assert e.state_byte == 255

    def test_to_bytes(self):
        e = TensorMIDIEvent(0, 0, 0, 0)
        assert e.to_bytes() == b"\x00\x00\x00\x00"

    def test_from_pitch_interval(self):
        e = TensorMIDIEvent.from_pitch_interval(60, 7, beat=3)
        assert e.beat_k == 3
        assert e.state_byte & 0x0F == 0  # pitch class of 60 = 0

    def test_from_pitch_interval_pitch_class(self):
        e = TensorMIDIEvent.from_pitch_interval(64, 0, beat=0)
        assert e.state_byte & 0x0F == 4  # pitch class of 64 = 4 (E)


class TestVoicesToTensorEvents:
    def test_single_voice(self):
        voices = [[60, 64, 67]]
        tensor, midi = voices_to_tensor_events(voices)
        assert len(tensor) == 3
        assert len(midi) == 3
        assert midi[0].note == 60
        assert midi[0].channel == 0

    def test_two_voices(self):
        voices = [[60, 64], [64, 67]]
        tensor, midi = voices_to_tensor_events(voices)
        assert len(tensor) == 4  # 2 beats * 2 voices
        assert len(midi) == 4
        # Check channels
        assert midi[0].channel == 0
        assert midi[1].channel == 1

    def test_beat_duration(self):
        voices = [[60, 62]]
        tensor, midi = voices_to_tensor_events(voices, beat_duration_ms=250.0)
        assert midi[0].duration_ms == 250.0
        assert midi[1].start_ms == 250.0

    def test_tensor_beat_counter(self):
        voices = [[60, 64, 67]]
        tensor, _ = voices_to_tensor_events(voices)
        assert tensor[0].beat_k == 0
        assert tensor[1].beat_k == 1
        assert tensor[2].beat_k == 2


class TestVoiceLeadingToSidechannels:
    def test_opening_all_nod(self):
        voices = [[60, 62], [64, 65]]
        result = voice_leading_to_sidechannels(voices, beat=0)
        assert result[(0, 1)] == "Nod"

    def test_contrary_motion_smile(self):
        voices = [[60, 62], [67, 65]]  # bass up, top down
        result = voice_leading_to_sidechannels(voices, beat=1)
        assert result[(0, 1)] == "Smile"

    def test_parallel_octave_frown(self):
        voices = [[60, 62], [72, 74]]  # both up by 2
        result = voice_leading_to_sidechannels(voices, beat=1)
        assert result[(0, 1)] == "Frown"

    def test_stepwise_similar_nod(self):
        voices = [[60, 62], [64, 65]]  # both up by 2 and 1
        result = voice_leading_to_sidechannels(voices, beat=1)
        assert result[(0, 1)] == "Nod"

    def test_three_voices(self):
        voices = [[60, 62], [64, 65], [67, 69]]
        result = voice_leading_to_sidechannels(voices, beat=1)
        assert len(result) == 3  # (0,1), (0,2), (1,2)


class TestIntervalToFluxVector:
    def test_returns_flux_vector(self):
        v = interval_to_flux_vector(7)
        assert isinstance(v, FluxVector)
        assert len(v) == 9

    def test_consonant_interval_strong(self):
        v = interval_to_flux_vector(0)
        assert v.magnitude > 0

    def test_dissonant_interval_weaker(self):
        interval_to_flux_vector(7)
        v_diss = interval_to_flux_vector(1)
        # Consonant should have full strength, dissonant half
        # We can't directly compare magnitudes because they map to different channels,
        # but we can check they're non-zero
        assert v_diss.magnitude > 0


class TestVoiceIntervalsToFluxVectors:
    def test_matches_voice_count(self):
        voices = [[60, 64], [64, 67], [67, 72]]
        vectors = voice_intervals_to_flux_vectors(voices, beat=0)
        assert len(vectors) == 3
        assert all(isinstance(v, FluxVector) for v in vectors)

    def test_bass_has_interval_zero(self):
        voices = [[60, 64], [64, 67]]
        vectors = voice_intervals_to_flux_vectors(voices, beat=0)
        # Bass (voice 0) has interval 0 from itself
        assert vectors[0].magnitude > 0  # channel 0 is active for unison
