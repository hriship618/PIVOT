import json
from pathlib import Path

from pivot.data.statsbomb import load_shots


ROOT = Path("models/data/raw/statsbomb")


with (ROOT / "matches.json").open("r", encoding="utf-8") as file:
    matches = json.load(file)


for match in matches:
    match_id = match["match_id"]

    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]

    home_score = match["home_score"]
    away_score = match["away_score"]

    official_goals = home_score + away_score

    shots = load_shots(
        ROOT / "events" / f"{match_id}.json",
        match_id,
    )

    parsed_goals = sum(shot.goal for shot in shots)

    if parsed_goals != official_goals:
        print(
            match_id,
            home,
            "vs",
            away,
            f"official={official_goals}",
            f"parsed={parsed_goals}",
        )