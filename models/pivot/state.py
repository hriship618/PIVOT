"""Tracking-state normalization, temporal encoding, targets, and dataset creation."""

from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pivot.data import (
    BallState,
    DFLShot,
    FrameState,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PlayerState,
    load_attacking_directions,
    load_dfl_goals,
    load_dfl_shots,
    load_metadata,
    load_tracking_frames,
    score_at_time,
)
from pivot.xg import (
    load_xg_model,
    predict_xg,
)


MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0

FUTURE_WINDOW_SECONDS = 15.0

# PIVOT observes the previous two seconds at 5 Hz.
TRACKING_HZ = 5.0
HISTORY_SECONDS = 2.0
HISTORY_FRAMES = int(
    TRACKING_HZ * HISTORY_SECONDS
)


def orient_player(
    player: PlayerState,
    attacking_right: bool,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> PlayerState:
    """Rotate one player so the focal team always attacks left-to-right."""

    if attacking_right:
        return deepcopy(player)

    return PlayerState(
        player_id=player.player_id,
        team_id=player.team_id,
        x=pitch_length - player.x,
        y=pitch_width - player.y,
        vx=-player.vx,
        vy=-player.vy,
    )


def orient_ball(
    ball: BallState,
    attacking_right: bool,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> BallState:
    """Rotate the ball into the focal team's attacking direction."""

    if attacking_right:
        return deepcopy(ball)

    return BallState(
        x=pitch_length - ball.x,
        y=pitch_width - ball.y,
        vx=-ball.vx,
        vy=-ball.vy,
    )


def orient_frame(
    frame: FrameState,
    attacking_right: bool,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> FrameState:
    """Normalize a complete frame so the focal team attacks left-to-right."""

    return replace(
        frame,
        players=[
            orient_player(
                player,
                attacking_right,
                pitch_length,
                pitch_width,
            )
            for player in frame.players
        ],
        ball=orient_ball(
            frame.ball,
            attacking_right,
            pitch_length,
            pitch_width,
        ),
    )


def players_in_encoding_order(
    frame: FrameState,
    focal_team_id: int | str,
    focal_attacking_right: bool,
) -> tuple[
    list[PlayerState],
    list[PlayerState],
]:
    """Return oriented focal/opponent players in the model's tensor order."""

    oriented = orient_frame(
        frame,
        focal_attacking_right,
    )

    focal = [
        player
        for player in oriented.players
        if player.team_id == focal_team_id
    ]

    opponents = [
        player
        for player in oriented.players
        if player.team_id != focal_team_id
    ]

    return focal, opponents


def _encode_player(
    player: PlayerState,
) -> list[float]:
    """Normalize one player's position and velocity."""

    return [
        player.x / PITCH_LENGTH,
        player.y / PITCH_WIDTH,
        player.vx / MAX_PLAYER_SPEED,
        player.vy / MAX_PLAYER_SPEED,
    ]


def _pad_players(
    players: list[PlayerState],
    size: int = 11,
) -> list[PlayerState]:
    """Pad a short-handed team so model tensors retain a fixed shape.

    The real manpower difference is separately included in match context,
    allowing red-card states to remain representable while keeping the neural
    network input shape fixed at 11 players per side.
    """

    if len(players) > size:
        raise ValueError(
            f"Expected at most {size} players, got {len(players)}"
        )

    padded = list(players)

    while len(padded) < size:
        padded.append(
            PlayerState(
                player_id="PAD",
                team_id="PAD",
                x=0.0,
                y=0.0,
                vx=0.0,
                vy=0.0,
            )
        )

    return padded


def encode_state(
    frame: FrameState,
    focal_team_id: int | str,
    focal_attacking_right: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Encode one tracking frame from the focal team's perspective."""

    focal_players, opponent_players = players_in_encoding_order(
        frame,
        focal_team_id,
        focal_attacking_right,
    )

    # Record real player counts before padding. This is what lets the network
    # distinguish 11v11 from 11v10, 10v10, etc.
    focal_player_count = len(
        focal_players
    )

    opponent_player_count = len(
        opponent_players
    )

    if (
        focal_player_count > 11
        or opponent_player_count > 11
    ):
        raise ValueError(
            "Expected at most 11 tracked players per team"
        )

    focal_players = _pad_players(
        focal_players
    )

    opponent_players = _pad_players(
        opponent_players
    )

    oriented = orient_frame(
        frame,
        focal_attacking_right,
    )

    focal_tensor = torch.tensor(
        [
            _encode_player(player)
            for player in focal_players
        ],
        dtype=torch.float32,
    )

    opponent_tensor = torch.tensor(
        [
            _encode_player(player)
            for player in opponent_players
        ],
        dtype=torch.float32,
    )

    ball = torch.tensor(
        [
            oriented.ball.x / PITCH_LENGTH,
            oriented.ball.y / PITCH_WIDTH,
            oriented.ball.vx / MAX_BALL_SPEED,
            oriented.ball.vy / MAX_BALL_SPEED,
        ],
        dtype=torch.float32,
    )

    if focal_team_id == oriented.home_team_id:
        score_difference = (
            oriented.home_score
            - oriented.away_score
        )
    else:
        score_difference = (
            oriented.away_score
            - oriented.home_score
        )

    player_count_difference = (
        focal_player_count
        - opponent_player_count
    )

    context = torch.tensor(
        [
            # Does the focal team currently have possession?
            float(
                oriented.possession_team_id
                == focal_team_id
            ),

            # Current game state.
            score_difference / 5.0,

            # Match time within the current period.
            oriented.timestamp
            / (45.0 * 60.0),

            # Crucial for distinguishing live play from corners/free kicks/etc.
            float(
                oriented.ball_in_play
            ),

            # Allows the model to learn the long-term cost of red cards.
            player_count_difference / 3.0,
        ],
        dtype=torch.float32,
    )

    return (
        focal_tensor,
        opponent_tensor,
        ball,
        context,
    )


def encode_sequence(
    frames: list[FrameState],
    focal_team_id: int | str,
    focal_attacking_right: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Encode a tracking history with an explicit time dimension."""

    encoded = [
        encode_state(
            frame,
            focal_team_id,
            focal_attacking_right,
        )
        for frame in frames
    ]

    focal = torch.stack(
        [
            state[0]
            for state in encoded
        ]
    )

    opponents = torch.stack(
        [
            state[1]
            for state in encoded
        ]
    )

    ball = torch.stack(
        [
            state[2]
            for state in encoded
        ]
    )

    context = torch.stack(
        [
            state[3]
            for state in encoded
        ]
    )

    return (
        focal,
        opponents,
        ball,
        context,
    )


def future_xg_target(
    frame: FrameState,
    shots: list[DFLShot],
    focal_team_id: str,
    window_seconds: float = FUTURE_WINDOW_SECONDS,
) -> float:
    """Compute future expected-goal differential following the current state."""

    value = 0.0

    end_time = (
        frame.timestamp
        + window_seconds
    )

    for shot in shots:
        if shot.period != frame.period:
            continue

        if shot.timestamp <= frame.timestamp:
            continue

        if shot.timestamp > end_time:
            continue

        if shot.team_id == focal_team_id:
            value += shot.xg

        elif shot.team_id is not None:
            value -= shot.xg

    return value


def _valid_history(
    frames: list[FrameState],
    end_index: int,
    history_frames: int,
    tracking_hz: float,
) -> list[FrameState] | None:
    """Return a contiguous tracking window ending at end_index.

    Dead-ball states are intentionally retained. Free kicks, corners, penalties
    and other restarts have expected-GD value and therefore belong in PIVOT.
    """

    start_index = (
        end_index
        - history_frames
        + 1
    )

    if start_index < 0:
        return None

    history = frames[
        start_index:
        end_index + 1
    ]

    current = history[-1]

    # Never allow a temporal sequence to cross a period boundary.
    if any(
        frame.period
        != current.period
        for frame in history
    ):
        return None

    # Missing a large number of players usually indicates bad tracking rather
    # than a legitimate card state. 20-22 players covers normal red-card cases.
    if any(
        len(frame.players) < 20
        for frame in history
    ):
        return None

    expected_interval = (
        1.0 / tracking_hz
    )

    maximum_gap = (
        expected_interval * 1.75
    )

    # Reject tracking discontinuities so the GRU receives a genuine short movie
    # rather than unrelated frames concatenated together.
    for previous, current_frame in zip(
        history,
        history[1:],
    ):
        gap = (
            current_frame.timestamp
            - previous.timestamp
        )

        if (
            gap <= 0
            or gap > maximum_gap
        ):
            return None

    return history


class StateDataset(Dataset):
    """Create temporal PIVOT training examples from synchronized tracking."""

    def __init__(
        self,
        match_dirs: list[str | Path],
        xg_checkpoint: str | Path,
        tracking_hz: float = TRACKING_HZ,
        sample_hz: float = 1.0,
        future_window: float = FUTURE_WINDOW_SECONDS,
        history_seconds: float = HISTORY_SECONDS,
    ) -> None:
        self.samples = []

        xg_model = load_xg_model(
            xg_checkpoint
        )

        xg_predictor = partial(
            predict_xg,
            xg_model,
        )

        history_frames = max(
            1,
            round(
                tracking_hz
                * history_seconds
            ),
        )

        sample_interval = (
            1.0 / sample_hz
        )

        for match_path in match_dirs:
            match_dir = Path(
                match_path
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
                xg_predictor,
            )

            goals = load_dfl_goals(
                match_dir
            )

            directions = load_attacking_directions(
                match_dir
            )

            home_team = metadata[
                "home_team_id"
            ]

            away_team = metadata[
                "away_team_id"
            ]

            # Score context must be correct at every frame in the temporal
            # sequence, not merely at its final frame.
            scored_frames = []

            for frame in frames:
                home_score, away_score = score_at_time(
                    goals,
                    frame.period,
                    frame.timestamp,
                    home_team,
                    away_team,
                )

                scored_frames.append(
                    replace(
                        frame,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )

            last_sample_time: dict[
                int,
                float,
            ] = {}

            for index, current_frame in enumerate(
                scored_frames
            ):
                history = _valid_history(
                    scored_frames,
                    index,
                    history_frames,
                    tracking_hz,
                )

                if history is None:
                    continue

                previous_sample = last_sample_time.get(
                    current_frame.period
                )

                if (
                    previous_sample is not None
                    and current_frame.timestamp
                    - previous_sample
                    < sample_interval
                ):
                    continue

                last_sample_time[
                    current_frame.period
                ] = current_frame.timestamp

                # Every state is learned from each team's perspective.
                for focal_team in (
                    home_team,
                    away_team,
                ):
                    direction_key = (
                        focal_team,
                        current_frame.period,
                    )

                    if direction_key not in directions:
                        continue

                    encoded = encode_sequence(
                        history,
                        focal_team,
                        directions[
                            direction_key
                        ],
                    )

                    target = future_xg_target(
                        current_frame,
                        shots,
                        focal_team,
                        window_seconds=future_window,
                    )

                    self.samples.append(
                        (
                            *encoded,
                            torch.tensor(
                                target,
                                dtype=torch.float32,
                            ),
                        )
                    )

    def __len__(
        self,
    ) -> int:
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