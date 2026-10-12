from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pivot.data.dfl import (
    load_attacking_directions,
    load_metadata,
    load_tracking_frames,
)

from pivot.data.dfl_events import (
    load_dfl_goals,
    load_dfl_shots,
    load_xg_model,
    score_at_time,
)

from pivot.data.state_targets import (
    future_xg_target,
)

from pivot.features.state_encoding import (
    encode_state,
)


class StateDataset(Dataset):

    def __init__(
        self,
        match_dirs: list[str | Path],
        xg_checkpoint: str | Path,
        tracking_hz: float = 5.0,
        sample_hz: float = 1.0,
        future_window: float = 15.0,
    ):

        self.samples = []

        xg_model = load_xg_model(
            xg_checkpoint
        )

        sample_interval = (
            1.0 / sample_hz
        )

        for match_dir in match_dirs:

            match_dir = Path(
                match_dir
            )

            print(
                "Loading",
                match_dir.name,
            )

            metadata = load_metadata(
                match_dir
            )

            frames = load_tracking_frames(
                match_dir,
                target_hz=tracking_hz,
            )

            shots = load_dfl_shots(
                match_dir,
                xg_model,
            )

            goals = load_dfl_goals(
                match_dir
            )

            directions = (
                load_attacking_directions(
                    match_dir
                )
            )

            home_team = metadata[
                "home_team_id"
            ]

            away_team = metadata[
                "away_team_id"
            ]

            last_sample_time = {}

            for frame in frames:

                if not frame.ball_in_play:
                    continue

                if len(frame.players) != 22:
                    continue

                # Downsample our training examples
                # to roughly sample_hz.
                previous_time = (
                    last_sample_time.get(
                        frame.period
                    )
                )

                if (
                    previous_time
                    is not None
                    and frame.timestamp
                    - previous_time
                    < sample_interval
                ):
                    continue

                last_sample_time[
                    frame.period
                ] = frame.timestamp

                (
                    home_score,
                    away_score,
                ) = score_at_time(
                    goals=goals,

                    period=frame.period,
                    timestamp=frame.timestamp,

                    home_team_id=home_team,
                    away_team_id=away_team,
                )

                scored_frame = replace(
                    frame,

                    home_score=home_score,
                    away_score=away_score,
                )

                # Make one training example
                # from each team's perspective.
                for focal_team in (
                    home_team,
                    away_team,
                ):

                    direction_key = (
                        focal_team,
                        frame.period,
                    )

                    if (
                        direction_key
                        not in directions
                    ):
                        continue

                    attacking_right = (
                        directions[
                            direction_key
                        ]
                    )

                    (
                        focal_players,
                        opponent_players,
                        ball,
                        context,
                    ) = encode_state(
                        frame=scored_frame,

                        focal_team_id=(
                            focal_team
                        ),

                        focal_attacking_right=(
                            attacking_right
                        ),
                    )

                    target = (
                        future_xg_target(
                            frame=scored_frame,

                            shots=shots,

                            focal_team_id=(
                                focal_team
                            ),

                            window_seconds=(
                                future_window
                            ),
                        )
                    )

                    self.samples.append(
                        (
                            focal_players,
                            opponent_players,
                            ball,
                            context,

                            torch.tensor(
                                target,
                                dtype=torch.float32,
                            ),
                        )
                    )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):
        return self.samples[
            index
        ]