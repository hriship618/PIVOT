from pivot.data.schema import (
    BallState,
    FrameState,
    PlayerState,
)

from pivot.features.state import (
    orient_frame,
)


def main():

    frame = FrameState(
        match_id=1,

        period=1,
        timestamp=100.0,

        home_team_id="home",
        away_team_id="away",

        possession_team_id="home",

        home_score=0,
        away_score=0,

        ball_in_play=True,

        players=[
            PlayerState(
                player_id="p1",

                team_id="home",

                x=80.0,
                y=20.0,

                vx=-4.0,
                vy=1.0,
            )
        ],

        ball=BallState(
            x=70.0,
            y=30.0,

            vx=-6.0,
            vy=2.0,
        ),
    )

    oriented = orient_frame(
        frame,
        attacking_right=False,
    )

    player = oriented.players[0]

    print(
        "Player:",
        player.x,
        player.y,
        player.vx,
        player.vy,
    )

    print(
        "Ball:",
        oriented.ball.x,
        oriented.ball.y,
        oriented.ball.vx,
        oriented.ball.vy,
    )


if __name__ == "__main__":
    main()