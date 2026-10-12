from pivot.data.schema import FrameState

from pivot.data.dfl_events import DFLShot


FUTURE_WINDOW_SECONDS = 15.0


def future_xg_target(
    frame: FrameState,
    shots: list[DFLShot],
    focal_team_id: str,
    window_seconds: float = FUTURE_WINDOW_SECONDS,
) -> float:

    start = frame.timestamp
    end = start + window_seconds

    value = 0.0

    for shot in shots:

        if shot.period != frame.period:
            continue

        if shot.timestamp <= start:
            continue

        if shot.timestamp > end:
            continue

        if shot.team_id == focal_team_id:
            value += shot.xg

        elif shot.team_id is not None:
            value -= shot.xg

    return value