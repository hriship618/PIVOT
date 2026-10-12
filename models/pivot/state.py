"""State normalization, tensor encoding, targets, and DFL training dataset."""

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
from pivot.xg import load_xg_model, predict_xg


MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0
FUTURE_WINDOW_SECONDS = 15.0


def orient_player(
    player: PlayerState,
    attacking_right: bool,
    pitch_length: float = PITCH_LENGTH,
    pitch_width: float = PITCH_WIDTH,
) -> PlayerState:
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
    """Rotate a frame 180 degrees when the focal team attacks left."""
    # Normalize both teams into the focal team's attacking direction so
    # equivalent spatial situations share the same representation.
    return replace(
        frame,
        players=[
            orient_player(player, attacking_right, pitch_length, pitch_width)
            for player in frame.players
        ],
        ball=orient_ball(frame.ball, attacking_right, pitch_length, pitch_width),
    )


def players_in_encoding_order(
    frame: FrameState,
    focal_team_id: int | str,
    focal_attacking_right: bool,
) -> tuple[list[PlayerState], list[PlayerState]]:
    """Return oriented players in the exact order used by tensor encoding."""
    oriented = orient_frame(frame, focal_attacking_right)
    focal = [player for player in oriented.players if player.team_id == focal_team_id]
    opponents = [player for player in oriented.players if player.team_id != focal_team_id]
    return focal, opponents


def _encode_player(player: PlayerState) -> list[float]:
    return [
        player.x / PITCH_LENGTH,
        player.y / PITCH_WIDTH,
        player.vx / MAX_PLAYER_SPEED,
        player.vy / MAX_PLAYER_SPEED,
    ]


def encode_state(
    frame: FrameState,
    focal_team_id: int | str,
    focal_attacking_right: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    focal_players, opponent_players = players_in_encoding_order(
        frame,
        focal_team_id,
        focal_attacking_right,
    )
    if len(focal_players) != 11 or len(opponent_players) != 11:
        raise ValueError("Expected 11 players for each team")

    oriented = orient_frame(frame, focal_attacking_right)
    focal_tensor = torch.tensor([_encode_player(player) for player in focal_players], dtype=torch.float32)
    opponent_tensor = torch.tensor(
        [_encode_player(player) for player in opponent_players], dtype=torch.float32
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
        score_difference = oriented.home_score - oriented.away_score
    else:
        score_difference = oriented.away_score - oriented.home_score
    context = torch.tensor(
        [
            float(oriented.possession_team_id == focal_team_id),
            score_difference / 5.0,
            oriented.timestamp / (45.0 * 60.0),
        ],
        dtype=torch.float32,
    )
    return focal_tensor, opponent_tensor, ball, context


def future_xg_target(
    frame: FrameState,
    shots: list[DFLShot],
    focal_team_id: str,
    window_seconds: float = FUTURE_WINDOW_SECONDS,
) -> float:
    value = 0.0
    end = frame.timestamp + window_seconds
    for shot in shots:
        if shot.period != frame.period or shot.timestamp <= frame.timestamp or shot.timestamp > end:
            continue
        if shot.team_id == focal_team_id:
            value += shot.xg
        elif shot.team_id is not None:
            value -= shot.xg
    return value


class StateDataset(Dataset):
    def __init__(
        self,
        match_dirs: list[str | Path],
        xg_checkpoint: str | Path,
        tracking_hz: float = 5.0,
        sample_hz: float = 1.0,
        future_window: float = FUTURE_WINDOW_SECONDS,
    ) -> None:
        self.samples = []
        xg_model = load_xg_model(xg_checkpoint)
        xg_predictor = partial(predict_xg, xg_model)
        sample_interval = 1.0 / sample_hz

        for match_path in match_dirs:
            match_dir = Path(match_path)
            print("Loading", match_dir.name)
            metadata = load_metadata(match_dir)
            frames = load_tracking_frames(match_dir, target_hz=tracking_hz)
            shots = load_dfl_shots(match_dir, xg_predictor)
            goals = load_dfl_goals(match_dir)
            directions = load_attacking_directions(match_dir)
            home_team = metadata["home_team_id"]
            away_team = metadata["away_team_id"]
            last_sample_time: dict[int, float] = {}

            for frame in frames:
                if not frame.ball_in_play or len(frame.players) != 22:
                    continue
                previous_time = last_sample_time.get(frame.period)
                if previous_time is not None and frame.timestamp - previous_time < sample_interval:
                    continue
                last_sample_time[frame.period] = frame.timestamp
                home_score, away_score = score_at_time(
                    goals,
                    frame.period,
                    frame.timestamp,
                    home_team,
                    away_team,
                )
                scored_frame = replace(frame, home_score=home_score, away_score=away_score)

                # Each live state becomes one example from each team's perspective.
                for focal_team in (home_team, away_team):
                    direction_key = (focal_team, frame.period)
                    if direction_key not in directions:
                        continue
                    encoded = encode_state(scored_frame, focal_team, directions[direction_key])
                    target = future_xg_target(
                        scored_frame,
                        shots,
                        focal_team,
                        window_seconds=future_window,
                    )
                    self.samples.append((*encoded, torch.tensor(target, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]

