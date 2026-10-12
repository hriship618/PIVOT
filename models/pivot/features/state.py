from copy import deepcopy

from pivot.data.schema import (
    BallState,
    FrameState,
    PlayerState,
)


DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


def orient_player(
    player: PlayerState,
    attacking_right: bool,
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
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
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
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
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
) -> FrameState:

    players = [
        orient_player(
            player,
            attacking_right,
            pitch_length,
            pitch_width,
        )
        for player in frame.players
    ]

    ball = orient_ball(
        frame.ball,
        attacking_right,
        pitch_length,
        pitch_width,
    )

    return FrameState(
        match_id=frame.match_id,

        period=frame.period,
        timestamp=frame.timestamp,

        home_team_id=frame.home_team_id,
        away_team_id=frame.away_team_id,
        possession_team_id=frame.possession_team_id,

        home_score=frame.home_score,
        away_score=frame.away_score,

        ball_in_play=frame.ball_in_play,

        players=players,
        ball=ball,
    )