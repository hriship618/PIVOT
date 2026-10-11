#creating a dataset of shots for shot quality training.
import json
from pathlib import Path

from pivot.data.schema import Shot
from pivot.data.statsbomb import load_shots


def load_all_statsbomb_shots(root: str | Path) -> list[Shot]:
    root = Path(root)

    matches_path = root / "matches.json"
    events_dir = root / "events"

    with matches_path.open("r", encoding="utf-8") as file:
        matches = json.load(file)

    all_shots: list[Shot] = []

    for match in matches:
        match_id = int(match["match_id"])
        event_path = events_dir / f"{match_id}.json"

        if not event_path.exists():
            continue

        shots = load_shots(event_path, match_id)
        all_shots.extend(shots)

    return all_shots