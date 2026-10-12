from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from pivot.data.dfl import (
    load_attacking_directions,
    load_metadata,
    load_period_starts,
)

from pivot.features.portable_xg import (
    shot_geometry_features,
)

from pivot.models.portable_xg import (
    PortableXGModel,
)


@dataclass
class DFLShot:
    match_id: str

    period: int
    timestamp: float

    team_id: str | None
    player_id: str | None

    x: float
    y: float

    penalty: bool
    goal: bool

    xg: float


@dataclass
class DFLGoal:
    period: int
    timestamp: float
    team_id: str | None


def _descendant(
    event: ET.Element,
    tag: str,
) -> ET.Element | None:

    return next(
        (
            node
            for node in event.iter()
            if node.tag == tag
        ),
        None,
    )


def _event_period_and_time(
    timestamp: datetime,
    period_starts: dict[int, datetime],
) -> tuple[int, float] | None:

    candidates = [
        (period, start)
        for period, start
        in period_starts.items()
        if timestamp >= start
    ]

    if not candidates:
        return None

    period, start = max(
        candidates,
        key=lambda item: item[1],
    )

    period_time = (
        timestamp - start
    ).total_seconds()

    return period, period_time


def load_xg_model(
    checkpoint_path: str | Path,
) -> PortableXGModel:

    model = PortableXGModel()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
        )
    )

    model.eval()

    return model


def predict_xg(
    model: PortableXGModel,
    x: float,
    y: float,
    penalty: bool,
) -> float:

    features = shot_geometry_features(
        x_m=x,
        y_m=y,
        penalty=penalty,
    )

    tensor = torch.tensor(
        features,
        dtype=torch.float32,
    )

    with torch.no_grad():

        logit = model(tensor)

        probability = torch.sigmoid(
            logit
        ).item()

    return probability


def load_dfl_shots(
    match_dir: str | Path,
    xg_model: PortableXGModel,
) -> list[DFLShot]:

    match_dir = Path(match_dir)

    metadata = load_metadata(
        match_dir
    )

    period_starts = load_period_starts(
        match_dir
    )

    attacking_directions = (
        load_attacking_directions(
            match_dir
        )
    )

    pitch_length = metadata[
        "pitch_length"
    ]

    pitch_width = metadata[
        "pitch_width"
    ]

    root = ET.parse(
        match_dir / "events.xml"
    ).getroot()

    shots = []

    for event in root.findall("Event"):

        shot = _descendant(
            event,
            "ShotAtGoal",
        )

        if shot is None:
            continue

        timestamp_raw = event.get(
            "EventTime"
        )

        if timestamp_raw is None:
            continue

        timestamp = datetime.fromisoformat(
            timestamp_raw
        )

        period_info = _event_period_and_time(
            timestamp,
            period_starts,
        )

        if period_info is None:
            continue

        period, period_time = period_info

        try:
            x = float(
                event.get(
                    "X-Source-Position"
                )
            )

            y = float(
                event.get(
                    "Y-Source-Position"
                )
            )

        except (TypeError, ValueError):
            continue

        team_id = shot.get("Team")
        player_id = shot.get("Player")

        attacking_right = (
            attacking_directions.get(
                (team_id, period),
                True,
            )
        )

        # Portable xG always expects the
        # shooting team to attack right.
        if not attacking_right:
            x = pitch_length - x
            y = pitch_width - y

        penalty = (
            _descendant(
                event,
                "Penalty",
            )
            is not None
        )

        goal = (
            _descendant(
                event,
                "SuccessfulShot",
            )
            is not None
        )

        xg = predict_xg(
            model=xg_model,
            x=x,
            y=y,
            penalty=penalty,
        )

        shots.append(
            DFLShot(
                match_id=metadata[
                    "match_id"
                ],

                period=period,
                timestamp=period_time,

                team_id=team_id,
                player_id=player_id,

                x=x,
                y=y,

                penalty=penalty,
                goal=goal,

                xg=xg,
            )
        )

    return shots


def load_dfl_goals(
    match_dir: str | Path,
) -> list[DFLGoal]:

    match_dir = Path(match_dir)

    period_starts = load_period_starts(
        match_dir
    )

    root = ET.parse(
        match_dir / "events.xml"
    ).getroot()

    goals = []

    for event in root.findall("Event"):

        shot = _descendant(
            event,
            "ShotAtGoal",
        )

        successful = _descendant(
            event,
            "SuccessfulShot",
        )

        if (
            shot is None
            or successful is None
        ):
            continue

        timestamp_raw = event.get(
            "EventTime"
        )

        if timestamp_raw is None:
            continue

        timestamp = datetime.fromisoformat(
            timestamp_raw
        )

        period_info = _event_period_and_time(
            timestamp,
            period_starts,
        )

        if period_info is None:
            continue

        period, period_time = period_info

        goals.append(
            DFLGoal(
                period=period,
                timestamp=period_time,
                team_id=shot.get("Team"),
            )
        )

    return goals


def score_at_time(
    goals: list[DFLGoal],
    period: int,
    timestamp: float,
    home_team_id: str,
    away_team_id: str,
) -> tuple[int, int]:

    home_score = 0
    away_score = 0

    for goal in goals:

        happened = (
            goal.period < period
            or (
                goal.period == period
                and goal.timestamp
                <= timestamp
            )
        )

        if not happened:
            continue

        if goal.team_id == home_team_id:
            home_score += 1

        elif goal.team_id == away_team_id:
            away_score += 1

    return (
        home_score,
        away_score,
    )