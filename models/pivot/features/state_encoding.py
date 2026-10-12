import torch

from pivot.data.schema import (
    FrameState,
)

from pivot.features.state import (
    orient_frame,
)


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0


def encode_player(
    player,
) -> list[float]:

    return [
        player.x / PITCH_LENGTH,
        player.y / PITCH_WIDTH,

        player.vx / MAX_PLAYER_SPEED,
        player.vy / MAX_PLAYER_SPEED,
    ]


def encode_state(
    frame: FrameState,
    focal_team_id: str,
    focal_attacking_right: bool,
):

    oriented = orient_frame(
        frame,
        attacking_right=(
            focal_attacking_right
        ),
    )

    focal_players = []

    opponent_players = []

    for player in oriented.players:

        encoded = encode_player(
            player
        )

        if (
            player.team_id
            == focal_team_id
        ):

            focal_players.append(
                encoded
            )

        else:
            opponent_players.append(
                encoded
            )

    if (
        len(focal_players) != 11
        or len(opponent_players) != 11
    ):
        raise ValueError(
            "Expected 11 players "
            "for each team"
        )

    focal_tensor = torch.tensor(
        focal_players,
        dtype=torch.float32,
    )

    opponent_tensor = torch.tensor(
        opponent_players,
        dtype=torch.float32,
    )

    ball = torch.tensor(
        [
            oriented.ball.x
            / PITCH_LENGTH,

            oriented.ball.y
            / PITCH_WIDTH,

            oriented.ball.vx
            / MAX_BALL_SPEED,

            oriented.ball.vy
            / MAX_BALL_SPEED,
        ],
        dtype=torch.float32,
    )

    possession = float(
        oriented.possession_team_id
        == focal_team_id
    )

    score_difference = 0.0

    if (
        focal_team_id
        == oriented.home_team_id
    ):

        score_difference = (
            oriented.home_score
            - oriented.away_score
        )

    else:

        score_difference = (
            oriented.away_score
            - oriented.home_score
        )

    context = torch.tensor(
        [
            possession,
            score_difference / 5.0,
            oriented.timestamp
            / (45.0 * 60.0),
        ],
        dtype=torch.float32,
    )

    return (
        focal_tensor,
        opponent_tensor,
        ball,
        context,
    )