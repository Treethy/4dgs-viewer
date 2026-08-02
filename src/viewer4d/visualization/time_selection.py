from __future__ import annotations

from dataclasses import dataclass

from viewer4d.core.model import GaussianFrame, SequenceInfo, Viewer4DGS


@dataclass(frozen=True, slots=True)
class TimeSelection:
    """Resolved time selection shared by both visualization modes."""

    time: float
    frame: int
    specified_by: str


def resolve_time_selection(
    sequence: SequenceInfo,
    *,
    time: float | None = None,
    frame: int | None = None,
) -> TimeSelection:
    """Resolve mutually exclusive normalized-time and frame inputs.

    When neither input is supplied, the first frame is selected. For a time
    input, ``frame`` stores the nearest discrete frame for display purposes;
    evaluation still uses the exact requested normalized time.
    """

    if time is not None and frame is not None:
        raise ValueError("time and frame are mutually exclusive")

    if frame is not None:
        resolved_time = sequence.frame_to_time(int(frame))
        return TimeSelection(
            time=resolved_time,
            frame=int(frame),
            specified_by="frame",
        )

    if time is None:
        return TimeSelection(
            time=sequence.time_min,
            frame=0,
            specified_by="default",
        )

    resolved_time = float(time)
    if not sequence.time_min <= resolved_time <= sequence.time_max:
        raise ValueError(
            f"time must be in [{sequence.time_min}, {sequence.time_max}], "
            f"got {resolved_time}"
        )
    if sequence.num_frames == 1 or sequence.time_max == sequence.time_min:
        nearest_frame = 0
    else:
        ratio = (resolved_time - sequence.time_min) / (
            sequence.time_max - sequence.time_min
        )
        nearest_frame = int(round(ratio * (sequence.num_frames - 1)))
        nearest_frame = max(0, min(sequence.num_frames - 1, nearest_frame))

    return TimeSelection(
        time=resolved_time,
        frame=nearest_frame,
        specified_by="time",
    )


def evaluate_selection(
    model: Viewer4DGS,
    selection: TimeSelection,
) -> GaussianFrame:
    """Evaluate a model without losing an exact fractional time selection."""

    if selection.specified_by == "frame":
        return model.evaluate_frame(selection.frame)
    return model.evaluate_time(selection.time)
