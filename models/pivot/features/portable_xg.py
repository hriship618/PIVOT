from math import atan2, pi, sqrt


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

GOAL_CENTER_Y = PITCH_WIDTH / 2
GOAL_WIDTH = 7.32


def shot_geometry_features(
    x_m: float,
    y_m: float,
    penalty: bool,
) -> list[float]:

    dx = PITCH_LENGTH - x_m
    dy = GOAL_CENTER_Y - y_m

    distance = sqrt(
        dx * dx + dy * dy
    )

    left_post_y = (
        GOAL_CENTER_Y
        - GOAL_WIDTH / 2
    )

    right_post_y = (
        GOAL_CENTER_Y
        + GOAL_WIDTH / 2
    )

    angle_left = atan2(
        left_post_y - y_m,
        dx,
    )

    angle_right = atan2(
        right_post_y - y_m,
        dx,
    )

    angle = abs(
        angle_right - angle_left
    )

    return [
        x_m / PITCH_LENGTH,
        y_m / PITCH_WIDTH,
        distance / PITCH_LENGTH,
        angle / pi,
        float(penalty),
    ]


def statsbomb_geometry_features(
    x: float,
    y: float,
    penalty: bool,
) -> list[float]:

    # StatsBomb coordinates:
    # 120 x 80
    #
    # Convert them onto our physical
    # 105 x 68 pitch representation.

    x_m = (
        x / 120.0
        * PITCH_LENGTH
    )

    y_m = (
        y / 80.0
        * PITCH_WIDTH
    )

    return shot_geometry_features(
        x_m=x_m,
        y_m=y_m,
        penalty=penalty,
    )