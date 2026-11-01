"""Room names keep a fault spec parseable after the batch suffix; the call recorder lays both sides on one timeline."""
import wave

import numpy as np

from services.agent.faults import faults_from_room_name
from services.sim.naming import room_name
from services.sim.recorder import CallRecorder


def test_room_name_moves_the_fault_spec_after_the_batch_suffix():
    name = room_name("demo-timeout--fault-check_availability-timeout", "052301", 1)
    assert name == "demo-timeout-052301-1--fault-check_availability-timeout"
    assert faults_from_room_name(name) == [{"tool": "check_availability", "call_index": 1, "error": "timeout"}]
    assert room_name("r3-clean-0432", "043230", 3) == "r3-clean-0432-043230-3"


def test_recorder_aligns_caller_and_agent_on_one_timeline(tmp_path):
    rec = CallRecorder(sample_rate=1000)
    rec.add("caller", 1000, np.full(100, 1000, dtype=np.int16))  # 100 ms at t=1.0 s
    rec.add("agent", 1500, np.full(50, -1000, dtype=np.int16))  # 50 ms at t=1.5 s
    out = rec.render()
    assert out.shape == (550, 2)
    assert out[:100, 0].tolist() == [1000] * 100 and not out[100:, 0].any()
    assert not out[:500, 1].any() and out[500:550, 1].tolist() == [-1000] * 50
    path = tmp_path / "call.wav"
    rec.save(path)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, 1000, 550)


def test_recorder_keeps_consecutive_frames_contiguous_despite_timestamp_jitter():
    rec = CallRecorder(sample_rate=1000)
    rec.add("caller", 1000, np.ones(10, dtype=np.int16))
    rec.add("caller", 1012, np.ones(10, dtype=np.int16))  # 2 ms late: still placed right after the first frame
    rec.add("caller", 1500, np.ones(10, dtype=np.int16))  # a real gap: re-anchored on the clock
    out = rec.render()
    assert out[:20, 0].tolist() == [1] * 20 and not out[20:500, 0].any() and out[500:510, 0].tolist() == [1] * 10


def test_recorder_without_audio_renders_nothing():
    assert CallRecorder(sample_rate=48000).render().shape == (0, 2)
