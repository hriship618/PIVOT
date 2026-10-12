from collections import Counter

from pivot.data.dfl import (
    load_tracking_frames,
)


def main():

    match_dir = (
        "models/data/raw/dfl/J03WMX"
    )

    frames = load_tracking_frames(
        match_dir,
        target_hz=5.0,
    )

    print(
        "Number of frames:",
        len(frames),
    )

    if not frames:
        return

    first = frames[0]

    print()
    print("First frame")
    print("-----------")

    print(
        "Match ID:",
        first.match_id,
    )

    print(
        "Period:",
        first.period,
    )

    print(
        "Timestamp:",
        first.timestamp,
    )

    print(
        "Players:",
        len(first.players),
    )

    print(
        "Possession:",
        first.possession_team_id,
    )

    print(
        "Ball in play:",
        first.ball_in_play,
    )

    print()
    print("Ball")
    print("----")

    print(
        first.ball.x,
        first.ball.y,
        first.ball.vx,
        first.ball.vy,
    )

    print()
    print("First few players")
    print("-----------------")

    for player in first.players[:5]:

        print(
            player.player_id,
            player.team_id,
            player.x,
            player.y,
            player.vx,
            player.vy,
        )

    # ------------------------------------------------
    # Overall player counts
    # ------------------------------------------------

    player_counts = [
        len(frame.players)
        for frame in frames
    ]

    print()
    print("Player count stats")
    print("------------------")

    print(
        "Min:",
        min(player_counts),
    )

    print(
        "Max:",
        max(player_counts),
    )

    print(
        "Average:",
        sum(player_counts)
        / len(player_counts),
    )

    # ------------------------------------------------
    # Ball coordinate range
    # ------------------------------------------------

    ball_x = [
        frame.ball.x
        for frame in frames
    ]

    ball_y = [
        frame.ball.y
        for frame in frames
    ]

    print()
    print("Ball coordinate ranges")
    print("----------------------")

    print(
        "x:",
        min(ball_x),
        "to",
        max(ball_x),
    )

    print(
        "y:",
        min(ball_y),
        "to",
        max(ball_y),
    )

    # ------------------------------------------------
    # Live / dead ball
    # ------------------------------------------------

    live_frames = [
        frame
        for frame in frames
        if frame.ball_in_play
    ]

    dead_frames = [
        frame
        for frame in frames
        if not frame.ball_in_play
    ]

    print()
    print("Live/dead frames")
    print("----------------")

    print(
        "Live:",
        len(live_frames),
    )

    print(
        "Dead:",
        len(dead_frames),
    )

    # ------------------------------------------------
    # Player counts during live play
    # ------------------------------------------------

    live_player_counts = [
        len(frame.players)
        for frame in live_frames
    ]

    print()
    print("Live player count stats")
    print("-----------------------")

    if live_player_counts:

        print(
            "Min:",
            min(live_player_counts),
        )

        print(
            "Max:",
            max(live_player_counts),
        )

        print(
            "Average:",
            sum(live_player_counts)
            / len(live_player_counts),
        )

    count_distribution = Counter(
        live_player_counts
    )

    print()
    print("Live frames by player count")
    print("---------------------------")

    for count in sorted(
        count_distribution
    ):

        print(
            count,
            "players:",
            count_distribution[count],
            "frames",
        )

    # ------------------------------------------------
    # Player speed sanity check
    # ------------------------------------------------

    speeds = []

    for frame in live_frames:

        for player in frame.players:

            speed = (
                player.vx ** 2
                + player.vy ** 2
            ) ** 0.5

            speeds.append(speed)

    print()
    print("Player speed stats")
    print("------------------")

    if speeds:

        print(
            "Min:",
            min(speeds),
        )

        print(
            "Max:",
            max(speeds),
        )

        print(
            "Average:",
            sum(speeds)
            / len(speeds),
        )

    # ------------------------------------------------
    # Possession coverage
    # ------------------------------------------------

    possession_known = sum(
        frame.possession_team_id
        is not None

        for frame in live_frames
    )

    print()
    print("Possession")
    print("----------")

    print(
        "Known:",
        possession_known,
    )

    print(
        "Unknown:",
        len(live_frames)
        - possession_known,
    )


if __name__ == "__main__":
    main()