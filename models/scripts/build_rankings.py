"""Build frontend-compatible PIVOT ranking JSON and CSV from DFL tracking."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from pivot.attribution import player_counterfactual_value
from pivot.data import (
    load_attacking_directions,
    load_dfl_goals,
    load_metadata,
    load_tracking_frames,
    score_at_time,
)
from pivot.model import StateValueModel
from pivot.state import encode_state, players_in_encoding_order


DATA_DIR = MODELS_DIR / "data"
RANKING_FIELDS = (
    "rank",
    "player_id",
    "name",
    "team_id",
    "matches",
    "minutes_sampled",
    "spatial_value",
    "prototype_pivot",
)


def _complete_dfl_matches(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / filename).exists() for filename in ("match.xml", "events.xml", "positions.xml"))
    )


def _finalize_rankings(totals: dict) -> list[dict]:
    rankings = []
    for player_id, record in totals.items():
        if record["frames"] == 0:
            continue
        rankings.append(
            {
                "player_id": player_id,
                "name": record["name"],
                "team_id": record["team_id"],
                "matches": len(record["matches"]),
                "minutes_sampled": round(record["frames"] / 60.0, 1),
                "spatial_value": record["sum_delta"] / record["frames"],
            }
        )
    rankings.sort(key=lambda row: row["spatial_value"], reverse=True)
    if not rankings:
        return rankings

    values = [row["spatial_value"] for row in rankings]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    standard_deviation = variance**0.5
    for rank, row in enumerate(rankings, start=1):
        z_score = (
            (row["spatial_value"] - mean_value) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )
        # Preserve the prototype display scale consumed by the frontend.
        row["prototype_pivot"] = round(10.0 * z_score, 2)
        row["spatial_value"] = round(row["spatial_value"], 6)
        row["rank"] = rank
    return rankings


def export_rankings(rankings: list[dict], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "prototype_rankings.json"
    csv_path = output_dir / "prototype_rankings.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rankings, file, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        writer.writerows(rankings)
    return json_path, csv_path


def build_rankings() -> list[dict]:
    torch.manual_seed(42)
    model = StateValueModel()
    model.load_state_dict(
        torch.load(DATA_DIR / "processed" / "state_value" / "full_model.pt", map_location="cpu")
    )
    model.eval()
    totals = defaultdict(
        lambda: {"name": "", "team_id": "", "sum_delta": 0.0, "frames": 0, "matches": set()}
    )

    for match_dir in _complete_dfl_matches(DATA_DIR / "raw" / "dfl"):
        print("Attributing", match_dir.name)
        metadata = load_metadata(match_dir)
        frames = load_tracking_frames(match_dir, target_hz=5.0)
        directions = load_attacking_directions(match_dir)
        goals = load_dfl_goals(match_dir)
        home_team = metadata["home_team_id"]
        away_team = metadata["away_team_id"]
        last_sample_time: dict[int, float] = {}

        for frame in frames:
            if not frame.ball_in_play or len(frame.players) != 22:
                continue
            previous = last_sample_time.get(frame.period)
            if previous is not None and frame.timestamp - previous < 1.0:
                continue
            last_sample_time[frame.period] = frame.timestamp
            scores = score_at_time(
                goals,
                frame.period,
                frame.timestamp,
                home_team,
                away_team,
            )
            scored_frame = replace(frame, home_score=scores[0], away_score=scores[1])

            for focal_team in (home_team, away_team):
                direction_key = (focal_team, frame.period)
                if direction_key not in directions:
                    continue
                attacking_right = directions[direction_key]
                focal_tensor, opponent_tensor, ball, context = encode_state(
                    scored_frame,
                    focal_team,
                    attacking_right,
                )
                focal_players, _ = players_in_encoding_order(
                    scored_frame,
                    focal_team,
                    attacking_right,
                )
                for index, player in enumerate(focal_players):
                    delta = player_counterfactual_value(
                        model,
                        focal_tensor,
                        opponent_tensor,
                        ball,
                        context,
                        index,
                    )
                    record = totals[player.player_id]
                    player_info = metadata["players"].get(player.player_id, {})
                    record["name"] = player_info.get("name", player.player_id)
                    record["team_id"] = player.team_id
                    record["sum_delta"] += delta
                    record["frames"] += 1
                    record["matches"].add(metadata["match_id"])
    return _finalize_rankings(totals)


def main() -> None:
    rankings = build_rankings()
    json_path, csv_path = export_rankings(
        rankings,
        DATA_DIR / "processed" / "rankings",
    )
    for row in rankings[:20]:
        print(
            f"{row['rank']:2d}. {row['name']:<22} "
            f"PIVOT={row['prototype_pivot']:>6.2f} | "
            f"raw={row['spatial_value']:+.6f} | {row['minutes_sampled']:.1f} min"
        )
    print("Saved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()

