from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

from pivot.data.schema import (
    BallState,
    FrameState,
    PlayerState,
)


PERIOD_BY_SECTION = {
    "firstHalf": 1,
    "secondHalf": 2,
    "firstHalfExtraTime": 3,
    "secondHalfExtraTime": 4,
}


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(value)


def load_metadata(match_dir: str | Path) -> dict:
    match_dir = Path(match_dir)

    root = ET.parse(match_dir / "match.xml").getroot()

    general = root.find("./MatchInformation/General")
    environment = root.find("./MatchInformation/Environment")

    if general is None:
        raise ValueError("Could not find match metadata")

    home_team_id = general.get("HomeTeamId")
    away_team_id = general.get("GuestTeamId")

    pitch_length = 105.0
    pitch_width = 68.0

    if environment is not None:
        pitch_length = float(
            environment.get("PitchX", pitch_length)
        )

        pitch_width = float(
            environment.get("PitchY", pitch_width)
        )

    players = {}

    for team in root.findall("./MatchInformation/Teams/Team"):
        team_id = team.get("TeamId")

        for player in team.findall("./Players/Player"):
            player_id = player.get("PersonId")

            if player_id is None:
                continue

            players[player_id] = {
                "team_id": team_id,
                "name": player.get("Shortname", player_id),
            }

    return {
        "match_id": (
            general.get("MatchId", "")
            .replace("DFL-MAT-", "")
        ),

        "home_team_id": home_team_id,
        "away_team_id": away_team_id,

        "pitch_length": pitch_length,
        "pitch_width": pitch_width,

        "players": players,
    }


def load_period_starts(
    match_dir: str | Path,
) -> dict[int, datetime]:
    match_dir = Path(match_dir)

    root = ET.parse(match_dir / "events.xml").getroot()

    starts = {}

    for event in root.findall("Event"):
        timestamp = _parse_time(
            event.get("EventTime")
        )

        if timestamp is None:
            continue

        for node in event.iter():
            if node.tag not in {"KickOff", "Kickoff"}:
                continue

            section = node.get("GameSection")

            if section not in PERIOD_BY_SECTION:
                continue

            period = PERIOD_BY_SECTION[section]

            starts.setdefault(
                period,
                timestamp,
            )

    return starts


def load_attacking_directions(
    match_dir: str | Path,
) -> dict[tuple[str, int], bool]:
    """
    Return whether each team attacks toward increasing x.

    True  -> attacks right
    False -> attacks left
    """

    match_dir = Path(match_dir)

    root = ET.parse(match_dir / "events.xml").getroot()

    directions = {}

    for event in root.findall("Event"):
        for node in event.iter():
            if node.tag not in {"KickOff", "Kickoff"}:
                continue

            section = node.get("GameSection")

            if section not in PERIOD_BY_SECTION:
                continue

            period = PERIOD_BY_SECTION[section]

            team_left = node.get("TeamLeft")
            team_right = node.get("TeamRight")

            if team_left is not None:
                directions[(team_left, period)] = True

            if team_right is not None:
                directions[(team_right, period)] = False

    return directions


def _possession_team(
    possession_code: str | None,
    home_team_id: str,
    away_team_id: str,
) -> str | None:
    if possession_code == "1":
        return home_team_id

    if possession_code == "2":
        return away_team_id

    return None


def load_tracking_frames(
    match_dir: str | Path,
    target_hz: float = 5.0,
) -> list[FrameState]:

    match_dir = Path(match_dir)

    metadata = load_metadata(match_dir)
    period_starts = load_period_starts(match_dir)

    positions_path = match_dir / "positions.xml"

    players = metadata["players"]

    pitch_length = metadata["pitch_length"]
    pitch_width = metadata["pitch_width"]

    home_team_id = metadata["home_team_id"]
    away_team_id = metadata["away_team_id"]

    # ------------------------------------------------
    # Pass 1:
    # Use the BALL to define our canonical timeline.
    # ------------------------------------------------

    sampled_frames: dict[
        tuple[int, int],
        dict,
    ] = {}

    source_hz = 25.0
    stride = max(
        1,
        round(source_hz / target_hz),
    )

    for _, frame_set in ET.iterparse(
        positions_path,
        events=("end",),
    ):

        if frame_set.tag != "FrameSet":
            continue

        team_marker = frame_set.get("TeamId")

        is_ball = (
            team_marker is not None
            and team_marker.lower() == "ball"
        )

        if not is_ball:
            frame_set.clear()
            continue

        section = frame_set.get("GameSection")

        period = PERIOD_BY_SECTION.get(
            section or ""
        )

        if period is None:
            frame_set.clear()
            continue

        ball_frames = list(frame_set)

        # Infer source tracking frequency from
        # actual timestamps when possible.
        if len(ball_frames) >= 2:

            t0 = _parse_time(
                ball_frames[0].get("T")
            )

            t1 = _parse_time(
                ball_frames[1].get("T")
            )

            if t0 is not None and t1 is not None:

                dt = (
                    t1 - t0
                ).total_seconds()

                if dt > 0:
                    source_hz = 1.0 / dt

                    stride = max(
                        1,
                        round(
                            source_hz
                            / target_hz
                        ),
                    )

        previous_ball = None

        for frame in ball_frames:

            try:
                frame_number = int(
                    frame.get("N")
                )

                raw_x = float(
                    frame.get("X")
                )

                raw_y = float(
                    frame.get("Y")
                )

            except (TypeError, ValueError):
                continue

            # Every `stride`th ORIGINAL DFL frame.
            if frame_number % stride != 0:
                continue

            timestamp = _parse_time(
                frame.get("T")
            )

            if timestamp is None:
                continue

            x = raw_x + pitch_length / 2
            y = raw_y + pitch_width / 2

            vx = 0.0
            vy = 0.0

            if previous_ball is not None:

                (
                    previous_timestamp,
                    previous_x,
                    previous_y,
                    previous_frame_number,
                ) = previous_ball

                dt = (
                    timestamp
                    - previous_timestamp
                ).total_seconds()

                # Don't differentiate across a
                # large tracking discontinuity.
                if (
                    0 < dt
                    <= 2.5 / target_hz
                ):
                    vx = (
                        x - previous_x
                    ) / dt

                    vy = (
                        y - previous_y
                    ) / dt

            previous_ball = (
                timestamp,
                x,
                y,
                frame_number,
            )

            key = (
                period,
                frame_number,
            )

            sampled_frames[key] = {
                "timestamp": timestamp,

                "ball": BallState(
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                ),

                "players": {},

                "possession": (
                    _possession_team(
                        frame.get(
                            "BallPossession"
                        ),
                        home_team_id,
                        away_team_id,
                    )
                ),

                "ball_in_play": (
                    frame.get(
                        "BallStatus"
                    )
                    == "1"
                ),
            }

        frame_set.clear()

    # ------------------------------------------------
    # Pass 2:
    # Read player positions ONLY for the exact
    # canonical frame numbers selected above.
    # ------------------------------------------------

    previous_positions = {}

    for _, frame_set in ET.iterparse(
        positions_path,
        events=("end",),
    ):

        if frame_set.tag != "FrameSet":
            continue

        person_id = frame_set.get(
            "PersonId"
        )

        if person_id is None:
            frame_set.clear()
            continue

        player_info = players.get(
            person_id
        )

        if player_info is None:
            frame_set.clear()
            continue

        section = frame_set.get(
            "GameSection"
        )

        period = PERIOD_BY_SECTION.get(
            section or ""
        )

        if period is None:
            frame_set.clear()
            continue

        for frame in frame_set:

            try:
                frame_number = int(
                    frame.get("N")
                )

            except (TypeError, ValueError):
                continue

            key = (
                period,
                frame_number,
            )

            # The ball did not select this frame,
            # so it is not part of our 5 Hz timeline.
            if key not in sampled_frames:
                continue

            try:
                raw_x = float(
                    frame.get("X")
                )

                raw_y = float(
                    frame.get("Y")
                )

            except (TypeError, ValueError):
                continue

            timestamp = _parse_time(
                frame.get("T")
            )

            if timestamp is None:
                continue

            x = raw_x + pitch_length / 2
            y = raw_y + pitch_width / 2

            vx = 0.0
            vy = 0.0

            velocity_key = (
                person_id,
                period,
            )

            previous = (
                previous_positions.get(
                    velocity_key
                )
            )

            if previous is not None:

                (
                    previous_timestamp,
                    previous_x,
                    previous_y,
                    previous_frame_number,
                ) = previous

                dt = (
                    timestamp
                    - previous_timestamp
                ).total_seconds()

                if (
                    0 < dt
                    <= 2.5 / target_hz
                ):
                    vx = (
                        x - previous_x
                    ) / dt

                    vy = (
                        y - previous_y
                    ) / dt

            previous_positions[
                velocity_key
            ] = (
                timestamp,
                x,
                y,
                frame_number,
            )

            sampled_frames[
                key
            ]["players"][
                person_id
            ] = PlayerState(
                player_id=person_id,

                team_id=player_info[
                    "team_id"
                ],

                x=x,
                y=y,

                vx=vx,
                vy=vy,
            )

        frame_set.clear()

    # ------------------------------------------------
    # Pass 3:
    # Convert synchronized records into FrameState.
    # ------------------------------------------------

    output = []

    for (
        period,
        frame_number,
    ), data in sorted(
        sampled_frames.items()
    ):

        timestamp = data["timestamp"]

        period_start = period_starts.get(
            period
        )

        if period_start is None:
            continue

        period_time = (
            timestamp
            - period_start
        ).total_seconds()

        output.append(
            FrameState(
                match_id=metadata[
                    "match_id"
                ],

                period=period,
                timestamp=period_time,

                home_team_id=(
                    home_team_id
                ),

                away_team_id=(
                    away_team_id
                ),

                possession_team_id=(
                    data["possession"]
                ),

                home_score=0,
                away_score=0,

                ball_in_play=(
                    data[
                        "ball_in_play"
                    ]
                ),

                players=list(
                    data[
                        "players"
                    ].values()
                ),

                ball=data["ball"],
            )
        )

    return output