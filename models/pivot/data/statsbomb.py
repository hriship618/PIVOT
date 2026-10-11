import json
from pathlib import Path

from pivot.data.schema import Shot


def load_events(path: str | Path) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def event_to_shot(event: dict, match_id: int) -> Shot | None:
    event_type = event.get("type", {}).get("name")

    if event_type != "Shot":
        return None

    period = int(event.get("period", 0))

    # Period 5 is a penalty shootout.
    # We do not want shootout attempts in our normal match xG model.
    if period > 4:
        return None

    location = event.get("location")

    if location is None or len(location) < 2:
        return None

    shot_data = event.get("shot", {})
    player_data = event.get("player", {})
    team_data = event.get("team", {})

    outcome = shot_data.get("outcome", {}).get("name")

    return Shot(
        match_id=match_id,

        player_id=int(player_data.get("id", -1)),
        team_id=int(team_data.get("id", -1)),

        period=period,
        timestamp=timestamp_to_seconds(
            event.get("timestamp", "00:00:00.000")
        ),

        x=float(location[0]),
        y=float(location[1]),

        body_part=shot_data.get(
            "body_part", {}
        ).get("name", "Unknown"),

        technique=shot_data.get(
            "technique", {}
        ).get("name", "Unknown"),

        shot_type=shot_data.get(
            "type", {}
        ).get("name", "Unknown"),

        under_pressure=bool(
            event.get("under_pressure", False)
        ),

        one_on_one=bool(
            shot_data.get("one_on_one", False)
        ),

        first_time=bool(
            shot_data.get("first_time", False)
        ),

        goal=outcome == "Goal",
    )


def load_shots(path: str | Path, match_id: int) -> list[Shot]:
    events = load_events(path)

    shots: list[Shot] = []

    for event in events:
        shot = event_to_shot(event, match_id)

        if shot is not None:
            shots.append(shot)

    return shots