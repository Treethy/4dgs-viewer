import pytest

from viewer4d import SequenceInfo
from viewer4d.visualization.time_selection import resolve_time_selection


def test_resolve_frame_and_time():
    sequence = SequenceInfo(num_frames=5, fps=20.0)
    by_frame = resolve_time_selection(sequence, frame=2)
    assert by_frame.time == pytest.approx(0.5)
    assert by_frame.frame == 2
    assert by_frame.specified_by == "frame"

    by_time = resolve_time_selection(sequence, time=0.74)
    assert by_time.time == pytest.approx(0.74)
    assert by_time.frame == 3
    assert by_time.specified_by == "time"


def test_time_defaults_and_exclusivity():
    sequence = SequenceInfo(num_frames=3, fps=12.0)
    default = resolve_time_selection(sequence)
    assert default.time == 0.0
    assert default.frame == 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_time_selection(sequence, time=0.5, frame=1)
