"""Schemas and parsers for StatsBomb events and DFL tracking/event data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Callable
from pivot.accounting import ActionEvent, ActionType

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
STATSBOMB_PITCH_LENGTH = 120.0
STATSBOMB_PITCH_WIDTH = 80.0

PERIOD_BY_SECTION = {
    "firstHalf": 1,
    "secondHalf": 2,
    "firstHalfExtraTime": 3,
    "secondHalfExtraTime": 4,
}


@dataclass
class PlayerState:
    player_id: int | str
    team_id: int | str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class BallState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class FrameState:
    match_id: int | str
    period: int
    timestamp: float
    home_team_id: int | str
    away_team_id: int | str
    possession_team_id: int | str | None
    home_score: int
    away_score: int
    ball_in_play: bool
    players: list[PlayerState]
    ball: BallState


@dataclass
class Shot:
    match_id: int
    player_id: int
    team_id: int
    period: int
    timestamp: float
    x: float
    y: float
    body_part: str
    technique: str
    shot_type: str
    under_pressure: bool
    one_on_one: bool
    first_time: bool
    statsbomb_xg: float
    goal: bool


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


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_events(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def event_to_shot(event: dict, match_id: int) -> Shot | None:
    if event.get("type", {}).get("name") != "Shot":
        return None

    period = int(event.get("period", 0))
    # Period 5 is a shootout, not part of normal-match xG training.
    if period > 4:
        return None

    location = event.get("location")
    if location is None or len(location) < 2:
        return None

    shot = event.get("shot", {})
    player = event.get("player", {})
    team = event.get("team", {})

    return Shot(
        match_id=match_id,
        player_id=int(player.get("id", -1)),
        team_id=int(team.get("id", -1)),
        period=period,
        timestamp=timestamp_to_seconds(event.get("timestamp", "00:00:00.000")),
        x=float(location[0]),
        y=float(location[1]),
        body_part=shot.get("body_part", {}).get("name", "Unknown"),
        technique=shot.get("technique", {}).get("name", "Unknown"),
        shot_type=shot.get("type", {}).get("name", "Unknown"),
        under_pressure=bool(event.get("under_pressure", False)),
        one_on_one=bool(shot.get("one_on_one", False)),
        first_time=bool(shot.get("first_time", False)),
        statsbomb_xg=float(shot.get("statsbomb_xg", 0.0)),
        goal=shot.get("outcome", {}).get("name") == "Goal",
    )


def load_shots(path: str | Path, match_id: int) -> list[Shot]:
    return [
        shot
        for event in load_events(path)
        if (shot := event_to_shot(event, match_id)) is not None
    ]


def load_all_statsbomb_shots(root: str | Path) -> list[Shot]:
    root = Path(root)
    with (root / "matches.json").open("r", encoding="utf-8") as file:
        matches = json.load(file)

    shots: list[Shot] = []
    for match in matches:
        match_id = int(match["match_id"])
        event_path = root / "events" / f"{match_id}.json"
        if event_path.exists():
            shots.extend(load_shots(event_path, match_id))
    return shots


def split_shots_chronologically(
    shots: list[Shot],
    statsbomb_root: str | Path,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[list[Shot], list[Shot], list[Shot]]:
    with (Path(statsbomb_root) / "matches.json").open("r", encoding="utf-8") as file:
        matches = json.load(file)

    ordered = sorted(
        matches,
        key=lambda match: (match["match_date"], match["kick_off"], match["match_id"]),
    )
    match_ids = [int(match["match_id"]) for match in ordered]
    train_end = int(len(match_ids) * train_fraction)
    validation_end = int(len(match_ids) * (train_fraction + validation_fraction))
    partitions = (
        set(match_ids[:train_end]),
        set(match_ids[train_end:validation_end]),
        set(match_ids[validation_end:]),
    )
    return tuple(
        [shot for shot in shots if shot.match_id in ids]
        for ids in partitions
    )


def load_metadata(match_dir: str | Path) -> dict:
    root = ET.parse(Path(match_dir) / "match.xml").getroot()
    general = root.find("./MatchInformation/General")
    environment = root.find("./MatchInformation/Environment")
    if general is None:
        raise ValueError("Could not find match metadata")

    pitch_length = PITCH_LENGTH
    pitch_width = PITCH_WIDTH
    if environment is not None:
        pitch_length = float(environment.get("PitchX", pitch_length))
        pitch_width = float(environment.get("PitchY", pitch_width))

    players = {}
    for team in root.findall("./MatchInformation/Teams/Team"):
        team_id = team.get("TeamId")
        for player in team.findall("./Players/Player"):
            player_id = player.get("PersonId")
            if player_id is not None:
                players[player_id] = {
                    "team_id": team_id,
                    "name": player.get("Shortname", player_id),
                }

    return {
        "match_id": general.get("MatchId", "").replace("DFL-MAT-", ""),
        "home_team_id": general.get("HomeTeamId"),
        "away_team_id": general.get("GuestTeamId"),
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "players": players,
    }


def load_period_starts(match_dir: str | Path) -> dict[int, datetime]:
    root = ET.parse(Path(match_dir) / "events.xml").getroot()
    starts: dict[int, datetime] = {}
    for event in root.findall("Event"):
        timestamp = _parse_time(event.get("EventTime"))
        if timestamp is None:
            continue
        for node in event.iter():
            if node.tag in {"KickOff", "Kickoff"}:
                period = PERIOD_BY_SECTION.get(node.get("GameSection", ""))
                if period is not None:
                    starts.setdefault(period, timestamp)
    return starts


def load_attacking_directions(match_dir: str | Path) -> dict[tuple[str, int], bool]:
    """Return whether each team attacks toward increasing x in each period."""
    root = ET.parse(Path(match_dir) / "events.xml").getroot()
    directions: dict[tuple[str, int], bool] = {}
    for event in root.findall("Event"):
        for node in event.iter():
            if node.tag not in {"KickOff", "Kickoff"}:
                continue
            period = PERIOD_BY_SECTION.get(node.get("GameSection", ""))
            if period is None:
                continue
            if (team_left := node.get("TeamLeft")) is not None:
                directions[(team_left, period)] = True
            if (team_right := node.get("TeamRight")) is not None:
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
    """Load synchronized DFL ball/player states on a ball-defined timeline."""
    match_dir = Path(match_dir)
    metadata = load_metadata(match_dir)
    period_starts = load_period_starts(match_dir)
    positions_path = match_dir / "positions.xml"
    pitch_length = metadata["pitch_length"]
    pitch_width = metadata["pitch_width"]
    home_team_id = metadata["home_team_id"]
    away_team_id = metadata["away_team_id"]
    sampled_frames: dict[tuple[int, int], dict] = {}

    source_hz = 25.0
    stride = max(1, round(source_hz / target_hz))

    # The ball defines the canonical frame numbers so every entity is synchronized.
    for _, frame_set in ET.iterparse(positions_path, events=("end",)):
        if frame_set.tag != "FrameSet":
            continue
        marker = frame_set.get("TeamId")
        if marker is None or marker.lower() != "ball":
            frame_set.clear()
            continue
        period = PERIOD_BY_SECTION.get(frame_set.get("GameSection", ""))
        if period is None:
            frame_set.clear()
            continue

        ball_frames = list(frame_set)
        if len(ball_frames) >= 2:
            t0 = _parse_time(ball_frames[0].get("T"))
            t1 = _parse_time(ball_frames[1].get("T"))
            if t0 is not None and t1 is not None and (dt := (t1 - t0).total_seconds()) > 0:
                source_hz = 1.0 / dt
                stride = max(1, round(source_hz / target_hz))

        previous_ball: tuple[datetime, float, float] | None = None
        for frame in ball_frames:
            try:
                frame_number = int(frame.get("N"))
                raw_x = float(frame.get("X"))
                raw_y = float(frame.get("Y"))
            except (TypeError, ValueError):
                continue
            if frame_number % stride != 0:
                continue
            timestamp = _parse_time(frame.get("T"))
            if timestamp is None:
                continue

            x = raw_x + pitch_length / 2
            y = raw_y + pitch_width / 2
            vx = vy = 0.0
            if previous_ball is not None:
                previous_time, previous_x, previous_y = previous_ball
                dt = (timestamp - previous_time).total_seconds()
                # Avoid differentiating across a tracking discontinuity.
                if 0 < dt <= 2.5 / target_hz:
                    vx = (x - previous_x) / dt
                    vy = (y - previous_y) / dt
            previous_ball = (timestamp, x, y)

            sampled_frames[(period, frame_number)] = {
                "timestamp": timestamp,
                "ball": BallState(x=x, y=y, vx=vx, vy=vy),
                "players": {},
                "possession": _possession_team(
                    frame.get("BallPossession"), home_team_id, away_team_id
                ),
                "ball_in_play": frame.get("BallStatus") == "1",
            }
        frame_set.clear()

    previous_positions: dict[tuple[str, int], tuple[datetime, float, float]] = {}
    for _, frame_set in ET.iterparse(positions_path, events=("end",)):
        if frame_set.tag != "FrameSet":
            continue
        person_id = frame_set.get("PersonId")
        player_info = metadata["players"].get(person_id)
        period = PERIOD_BY_SECTION.get(frame_set.get("GameSection", ""))
        if person_id is None or player_info is None or period is None:
            frame_set.clear()
            continue

        for frame in frame_set:
            try:
                frame_number = int(frame.get("N"))
            except (TypeError, ValueError):
                continue
            key = (period, frame_number)
            if key not in sampled_frames:
                continue
            try:
                raw_x = float(frame.get("X"))
                raw_y = float(frame.get("Y"))
            except (TypeError, ValueError):
                continue
            timestamp = _parse_time(frame.get("T"))
            if timestamp is None:
                continue

            x = raw_x + pitch_length / 2
            y = raw_y + pitch_width / 2
            vx = vy = 0.0
            velocity_key = (person_id, period)
            if (previous := previous_positions.get(velocity_key)) is not None:
                previous_time, previous_x, previous_y = previous
                dt = (timestamp - previous_time).total_seconds()
                if 0 < dt <= 2.5 / target_hz:
                    vx = (x - previous_x) / dt
                    vy = (y - previous_y) / dt
            previous_positions[velocity_key] = (timestamp, x, y)
            sampled_frames[key]["players"][person_id] = PlayerState(
                player_id=person_id,
                team_id=player_info["team_id"],
                x=x,
                y=y,
                vx=vx,
                vy=vy,
            )
        frame_set.clear()

    frames = []
    for (period, _), data in sorted(sampled_frames.items()):
        period_start = period_starts.get(period)
        if period_start is None:
            continue
        frames.append(
            FrameState(
                match_id=metadata["match_id"],
                period=period,
                timestamp=(data["timestamp"] - period_start).total_seconds(),
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                possession_team_id=data["possession"],
                home_score=0,
                away_score=0,
                ball_in_play=data["ball_in_play"],
                players=list(data["players"].values()),
                ball=data["ball"],
            )
        )
    return frames


def _descendant(event: ET.Element, tag: str) -> ET.Element | None:
    return next((node for node in event.iter() if node.tag == tag), None)


def _event_period_and_time(
    timestamp: datetime,
    period_starts: dict[int, datetime],
) -> tuple[int, float] | None:
    candidates = [(period, start) for period, start in period_starts.items() if timestamp >= start]
    if not candidates:
        return None
    period, start = max(candidates, key=lambda item: item[1])
    return period, (timestamp - start).total_seconds()


def load_dfl_shots(
    match_dir: str | Path,
    predict_xg: Callable[[float, float, bool], float],
) -> list[DFLShot]:
    match_dir = Path(match_dir)
    metadata = load_metadata(match_dir)
    period_starts = load_period_starts(match_dir)
    directions = load_attacking_directions(match_dir)
    root = ET.parse(match_dir / "events.xml").getroot()
    shots = []

    for event in root.findall("Event"):
        shot = _descendant(event, "ShotAtGoal")
        timestamp = _parse_time(event.get("EventTime"))
        if shot is None or timestamp is None:
            continue
        period_info = _event_period_and_time(timestamp, period_starts)
        if period_info is None:
            continue
        period, period_time = period_info
        try:
            x = float(event.get("X-Source-Position"))
            y = float(event.get("Y-Source-Position"))
        except (TypeError, ValueError):
            continue

        team_id = shot.get("Team")
        # xG is provider-neutral only after the shooting team attacks right.
        if not directions.get((team_id, period), True):
            x = metadata["pitch_length"] - x
            y = metadata["pitch_width"] - y
        penalty = _descendant(event, "Penalty") is not None
        shots.append(
            DFLShot(
                match_id=metadata["match_id"],
                period=period,
                timestamp=period_time,
                team_id=team_id,
                player_id=shot.get("Player"),
                x=x,
                y=y,
                penalty=penalty,
                goal=_descendant(event, "SuccessfulShot") is not None,
                xg=predict_xg(x, y, penalty),
            )
        )
    return shots


def load_dfl_goals(match_dir: str | Path) -> list[DFLGoal]:
    match_dir = Path(match_dir)
    period_starts = load_period_starts(match_dir)
    root = ET.parse(match_dir / "events.xml").getroot()
    goals = []
    for event in root.findall("Event"):
        shot = _descendant(event, "ShotAtGoal")
        timestamp = _parse_time(event.get("EventTime"))
        if shot is None or _descendant(event, "SuccessfulShot") is None or timestamp is None:
            continue
        period_info = _event_period_and_time(timestamp, period_starts)
        if period_info is not None:
            period, period_time = period_info
            goals.append(DFLGoal(period, period_time, shot.get("Team")))
    return goals


def score_at_time(
    goals: list[DFLGoal],
    period: int,
    timestamp: float,
    home_team_id: str,
    away_team_id: str,
) -> tuple[int, int]:
    home_score = away_score = 0
    for goal in goals:
        happened = goal.period < period or (
            goal.period == period and goal.timestamp <= timestamp
        )
        if happened and goal.team_id == home_team_id:
            home_score += 1
        elif happened and goal.team_id == away_team_id:
            away_score += 1
    return home_score, away_score

# ---------------------------------------------------------------------------
# Provider-neutral DFL action events
# ---------------------------------------------------------------------------


def _first_attribute(
    nodes: list[ET.Element],
    names: tuple[str, ...],
) -> str | None:
    """Return the first matching XML attribute from a group of nodes."""

    for node in nodes:
        for name in names:
            value = node.get(name)

            if value not in (None, ""):
                return value

    return None


def _optional_float(
    value: str | None,
) -> float | None:
    if value is None:
        return None

    try:
        return float(value)

    except ValueError:
        return None


def _xml_boolean(
    value: str | None,
) -> bool | None:
    """Interpret common provider boolean encodings."""

    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "successful",
        "success",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "unsuccessful",
        "failed",
        "failure",
    }:
        return False

    return None


def _event_actor(
    event: ET.Element,
    action_node: ET.Element,
) -> tuple[
    str | None,
    str | None,
]:
    """Extract team/player IDs without tying the model to one XML spelling."""

    nodes = [
        action_node,
        event,
    ]

    team_id = _first_attribute(
        nodes,
        (
            "Team",
            "TeamId",
            "TeamID",
        ),
    )

    player_id = _first_attribute(
        nodes,
        (
            "Player",
            "PlayerId",
            "PlayerID",
            "PersonId",
            "PersonID",
        ),
    )

    return team_id, player_id


def _event_opponent(
    event: ET.Element,
    action_node: ET.Element,
) -> tuple[
    str | None,
    str | None,
]:
    """Extract an explicitly named opposing team/player when available."""

    nodes = [
        action_node,
        event,
    ]

    opponent_team = _first_attribute(
        nodes,
        (
            "OpponentTeam",
            "OpponentTeamId",
            "OpponentTeamID",
        ),
    )

    opponent_player = _first_attribute(
        nodes,
        (
            "Opponent",
            "OpponentPlayer",
            "OpponentPlayerId",
            "OpponentPlayerID",
        ),
    )

    return (
        opponent_team,
        opponent_player,
    )


def _event_location(
    event: ET.Element,
    action_node: ET.Element,
) -> tuple[
    float | None,
    float | None,
]:
    """Read the event origin on the physical pitch when provided."""

    x = _first_attribute(
        [action_node, event],
        (
            "X-Source-Position",
            "X",
            "PosX",
        ),
    )

    y = _first_attribute(
        [action_node, event],
        (
            "Y-Source-Position",
            "Y",
            "PosY",
        ),
    )

    return (
        _optional_float(x),
        _optional_float(y),
    )


def _card_type(
    node: ET.Element,
) -> ActionType:
    """Interpret card events conservatively."""

    text = " ".join(
        str(value).lower()
        for value in node.attrib.values()
    )

    tag = node.tag.lower()

    if (
        "red" in text
        or "dismiss" in text
        or "sendingoff" in tag
        or "redcard" in tag
    ):
        return ActionType.RED_CARD

    return ActionType.YELLOW_CARD


def _classify_dfl_event(
    event: ET.Element,
) -> tuple[
    ActionType,
    ET.Element,
] | None:
    """Find the principal football action represented by one provider event.

    Order matters. A penalty shot, for example, is a SHOT rather than a second
    penalty-award event if the provider includes both ShotAtGoal and Penalty
    descendants.
    """

    nodes = list(
        event.iter()
    )

    tags = {
        node.tag: node
        for node in nodes
    }

    # Most specific events first.
    if "ShotAtGoal" in tags:
        return (
            ActionType.SHOT,
            tags["ShotAtGoal"],
        )

    for tag in (
        "RedCard",
        "SendingOff",
    ):
        if tag in tags:
            return (
                ActionType.RED_CARD,
                tags[tag],
            )

    if "Caution" in tags:
        return (
            _card_type(tags["Caution"]),
            tags["Caution"],
        )

    if "Penalty" in tags:
        return (
            ActionType.PENALTY_WON,
            tags["Penalty"],
        )

    if "CornerKick" in tags:
        return (
            ActionType.CORNER_WON,
            tags["CornerKick"],
        )

    if "FreeKick" in tags:
        return (
            ActionType.FREE_KICK_WON,
            tags["FreeKick"],
        )

    if "Foul" in tags:
        return (
            ActionType.FOUL_CONCEDED,
            tags["Foul"],
        )

    for tag in (
        "Interception",
        "InterceptBall",
    ):
        if tag in tags:
            return (
                ActionType.INTERCEPTION,
                tags[tag],
            )

    if "TacklingGame" in tags:
        return (
            ActionType.TACKLE,
            tags["TacklingGame"],
        )

    if "Clearance" in tags:
        return (
            ActionType.CLEARANCE,
            tags["Clearance"],
        )

    if "Cross" in tags:
        return (
            ActionType.CROSS,
            tags["Cross"],
        )

    if "Pass" in tags:
        return (
            ActionType.PASS,
            tags["Pass"],
        )

    # Some providers express ordinary ball actions under a generic Play node.
    if "Play" in tags:
        return (
            ActionType.OTHER,
            tags["Play"],
        )

    return None


def load_dfl_action_events(
    match_dir: str | Path,
    predict_xg: Callable[
        [float, float, bool],
        float,
    ] | None = None,
) -> list[ActionEvent]:
    """Normalize DFL event XML into provider-neutral PIVOT actions.

    Parsing is deliberately conservative: when the provider does not explicitly
    identify a beneficiary/player, that information remains None rather than
    being guessed.

    Carries and continuous off-ball movement are not fabricated here; those are
    derived from tracking rather than event labels.
    """

    match_dir = Path(
        match_dir
    )

    metadata = load_metadata(
        match_dir
    )

    period_starts = load_period_starts(
        match_dir
    )

    directions = load_attacking_directions(
        match_dir
    )

    root = ET.parse(
        match_dir / "events.xml"
    ).getroot()

    result: list[ActionEvent] = []

    for event in root.findall(
        "Event"
    ):
        event_time = _parse_time(
            event.get("EventTime")
        )

        if event_time is None:
            continue

        period_info = _event_period_and_time(
            event_time,
            period_starts,
        )

        if period_info is None:
            continue

        period, timestamp = period_info

        classified = _classify_dfl_event(
            event
        )

        if classified is None:
            continue

        event_type, action_node = classified

        team_id, player_id = _event_actor(
            event,
            action_node,
        )

        opponent_team, opponent_player = _event_opponent(
            event,
            action_node,
        )

        x, y = _event_location(
            event,
            action_node,
        )

        success = _xml_boolean(
            _first_attribute(
                [
                    action_node,
                    event,
                ],
                (
                    "Successful",
                    "Success",
                    "Outcome",
                ),
            )
        )

        goal: bool | None = None
        xg: float | None = None

        if event_type == ActionType.SHOT:
            goal = (
                _descendant(
                    event,
                    "SuccessfulShot",
                )
                is not None
            )

            penalty = (
                _descendant(
                    event,
                    "Penalty",
                )
                is not None
            )

            # xG expects the shooting team to attack left-to-right.
            if (
                x is not None
                and y is not None
                and team_id is not None
                and not directions.get(
                    (
                        team_id,
                        period,
                    ),
                    True,
                )
            ):
                x = (
                    metadata["pitch_length"]
                    - x
                )

                y = (
                    metadata["pitch_width"]
                    - y
                )

            if (
                predict_xg is not None
                and x is not None
                and y is not None
            ):
                xg = predict_xg(
                    x,
                    y,
                    penalty,
                )

        result.append(
            ActionEvent(
                match_id=metadata[
                    "match_id"
                ],
                period=period,
                timestamp=timestamp,
                event_type=event_type,
                team_id=team_id,
                player_id=player_id,
                x=x,
                y=y,
                opponent_team_id=opponent_team,
                opponent_player_id=opponent_player,
                successful=success,
                xg=xg,
                goal=goal,
                qualifier=action_node.tag,
                provider_event_id=event.get(
                    "EventId"
                )
                or event.get(
                    "EventID"
                )
                or event.get(
                    "Id"
                ),
            )
        )

    result.sort(
        key=lambda action: (
            action.period,
            action.timestamp,
        )
    )

    return result

def test_action_event_normalization(
    tmp_path: Path,
) -> None:
    _write_minimal_dfl_match(
        tmp_path
    )

    events_path = (
        tmp_path / "events.xml"
    )

    events_path.write_text(
        """
<Root>
    <Event EventId="kickoff" EventTime="2024-01-01T12:00:00">
        <KickOff
            GameSection="firstHalf"
            TeamLeft="home"
            TeamRight="away"
        />
    </Event>

    <Event
        EventId="tackle-1"
        EventTime="2024-01-01T12:00:05"
        X-Source-Position="50"
        Y-Source-Position="30"
    >
        <TacklingGame
            Team="home"
            Player="h4"
            Successful="1"
        />
    </Event>

    <Event
        EventId="foul-1"
        EventTime="2024-01-01T12:00:10"
        X-Source-Position="85"
        Y-Source-Position="34"
    >
        <Foul
            Team="away"
            Player="a3"
        />
    </Event>

    <Event
        EventId="shot-1"
        EventTime="2024-01-01T12:00:12"
        X-Source-Position="95"
        Y-Source-Position="34"
    >
        <ShotAtGoal
            Team="home"
            Player="h9"
        />
        <SuccessfulShot />
    </Event>
</Root>
""".strip(),
        encoding="utf-8",
    )

    actions = load_dfl_action_events(
        tmp_path
    )

    assert len(actions) == 3

    assert actions[0].event_type == ActionType.TACKLE
    assert actions[0].player_id == "h4"
    assert actions[0].successful is True

    assert actions[1].event_type == ActionType.FOUL_CONCEDED
    assert actions[1].player_id == "a3"

    assert actions[2].event_type == ActionType.SHOT
    assert actions[2].player_id == "h9"
    assert actions[2].goal is True

    assert actions[0].provider_event_id == "tackle-1"