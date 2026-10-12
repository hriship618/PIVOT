import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from pivot.data.dfl import (
    load_attacking_directions,
    load_metadata,
    load_tracking_frames,
)

from pivot.data.dfl_events import (
    load_dfl_goals,
    score_at_time,
)

from pivot.features.state import (
    orient_frame,
)

from pivot.features.state_encoding import (
    encode_state,
)

from pivot.models.state_value import (
    StateValueModel,
)

from pivot.attribution.counterfactual import (
    player_counterfactual_value,
)


def main():

    torch.manual_seed(42)

    dfl_root = Path(
        "models/data/raw/dfl"
    )

    match_dirs = sorted(
        path
        for path in dfl_root.iterdir()
        if (
            path.is_dir()
            and (path / "match.xml").exists()
            and (path / "events.xml").exists()
            and (path / "positions.xml").exists()
        )
    )

    model = StateValueModel()

    model.load_state_dict(
        torch.load(
            "models/data/processed/state_value/full_model.pt",
            map_location="cpu",
        )
    )

    model.eval()

    # player_id -> accumulated statistics
    totals = defaultdict(
        lambda: {
            "name": "",
            "team_id": "",
            "sum_delta": 0.0,
            "frames": 0,
            "matches": set(),
        }
    )

    for match_dir in match_dirs:

        print(
            "Attributing",
            match_dir.name,
        )

        metadata = load_metadata(
            match_dir
        )

        frames = load_tracking_frames(
            match_dir,
            target_hz=5.0,
        )

        directions = (
            load_attacking_directions(
                match_dir
            )
        )

        goals = load_dfl_goals(
            match_dir
        )

        home_team = metadata[
            "home_team_id"
        ]

        away_team = metadata[
            "away_team_id"
        ]

        # Only attribute approximately once
        # per second for this prototype.
        last_sample_time = {}

        for frame in frames:

            if not frame.ball_in_play:
                continue

            if len(frame.players) != 22:
                continue

            previous = last_sample_time.get(
                frame.period
            )

            if (
                previous is not None
                and frame.timestamp - previous < 1.0
            ):
                continue

            last_sample_time[
                frame.period
            ] = frame.timestamp

            home_score, away_score = (
                score_at_time(
                    goals=goals,
                    period=frame.period,
                    timestamp=frame.timestamp,
                    home_team_id=home_team,
                    away_team_id=away_team,
                )
            )

            # Update score directly for encoding.
            frame.home_score = home_score
            frame.away_score = away_score

            for focal_team in (
                home_team,
                away_team,
            ):

                direction_key = (
                    focal_team,
                    frame.period,
                )

                if direction_key not in directions:
                    continue

                attacking_right = directions[
                    direction_key
                ]

                (
                    focal_tensor,
                    opponent_tensor,
                    ball,
                    context,
                ) = encode_state(
                    frame=frame,
                    focal_team_id=focal_team,
                    focal_attacking_right=attacking_right,
                )

                # orient_frame preserves player order,
                # allowing us to associate tensor rows
                # with player IDs.
                oriented = orient_frame(
                    frame,
                    attacking_right=attacking_right,
                )

                focal_players = [
                    player
                    for player in oriented.players
                    if player.team_id == focal_team
                ]

                if len(focal_players) != 11:
                    continue

                for index, player in enumerate(
                    focal_players
                ):

                    delta = (
                        player_counterfactual_value(
                            model=model,

                            focal_players=focal_tensor,
                            opponent_players=opponent_tensor,

                            ball=ball,
                            context=context,

                            player_index=index,
                        )
                    )

                    record = totals[
                        player.player_id
                    ]

                    player_info = metadata[
                        "players"
                    ].get(
                        player.player_id,
                        {},
                    )

                    record["name"] = (
                        player_info.get(
                            "name",
                            player.player_id,
                        )
                    )

                    record["team_id"] = (
                        player.team_id
                    )

                    record[
                        "sum_delta"
                    ] += delta

                    record[
                        "frames"
                    ] += 1

                    record[
                        "matches"
                    ].add(
                        metadata[
                            "match_id"
                        ]
                    )

    rankings = []

    for player_id, record in totals.items():

        frames = record[
            "frames"
        ]

        if frames == 0:
            continue

        average_delta = (
            record["sum_delta"]
            / frames
        )

        minutes = (
            frames / 60.0
        )

        rankings.append(
            {
                "player_id": player_id,
                "name": record["name"],
                "team_id": record["team_id"],

                "matches": len(
                    record["matches"]
                ),

                "minutes_sampled": round(
                    minutes,
                    1,
                ),

                "spatial_value": (
                    average_delta
                ),
            }
        )

    rankings.sort(
        key=lambda row: row[
            "spatial_value"
        ],
        reverse=True,
    )

    # ------------------------------------------------
    # Give frontend a readable prototype scale.
    #
    # This is NOT final PIVOT.
    #
    # 0 = average player in this sample.
    # Positive = above sample average.
    # Negative = below.
    # ------------------------------------------------

    values = [
        row["spatial_value"]
        for row in rankings
    ]

    mean_value = (
        sum(values) / len(values)
    )

    variance = (
        sum(
            (value - mean_value) ** 2
            for value in values
        )
        / len(values)
    )

    std = variance ** 0.5

    for rank, row in enumerate(
        rankings,
        start=1,
    ):

        if std > 0:
            z = (
                row["spatial_value"]
                - mean_value
            ) / std
        else:
            z = 0.0

        # Temporary readable display score.
        row["prototype_pivot"] = round(
            10.0 * z,
            2,
        )

        row["spatial_value"] = round(
            row["spatial_value"],
            6,
        )

        row["rank"] = rank

    output_dir = Path(
        "models/data/processed/rankings"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "prototype_rankings.json"
    )

    csv_path = (
        output_dir
        / "prototype_rankings.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            rankings,
            file,
            indent=2,
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "player_id",
                "name",
                "team_id",
                "matches",
                "minutes_sampled",
                "spatial_value",
                "prototype_pivot",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rankings
        )

    print()
    print("Prototype PIVOT rankings")
    print("------------------------")

    for row in rankings[:20]:

        print(
            f"{row['rank']:2d}. "
            f"{row['name']:<22} "
            f"PIVOT={row['prototype_pivot']:>6.2f} | "
            f"raw={row['spatial_value']:+.6f} | "
            f"{row['minutes_sampled']:.1f} min"
        )

    print()
    print(
        "Saved:",
        json_path,
    )

    print(
        "Saved:",
        csv_path,
    )


if __name__ == "__main__":
    main()