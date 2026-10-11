#we use distance and angle and describe shots to train a shot quality model.

from math import atan2, sqrt

from pivot.data.schema import Shot


GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_WIDTH = 7.32


def shot_distance(shot: Shot) -> float:
    dx = GOAL_X - shot.x
    dy = GOAL_CENTER_Y - shot.y

    return sqrt(dx * dx + dy * dy)


def shot_angle(shot: Shot) -> float:
    left_post_y = GOAL_CENTER_Y - GOAL_WIDTH / 2
    right_post_y = GOAL_CENTER_Y + GOAL_WIDTH / 2

    dx = GOAL_X - shot.x

    angle_left = atan2(left_post_y - shot.y, dx)
    angle_right = atan2(right_post_y - shot.y, dx)

    return abs(angle_right - angle_left)


def shot_numeric_features(shot: Shot) -> list[float]:
    return [
        shot.x,
        shot.y,
        shot_distance(shot),
        shot_angle(shot),
        float(shot.under_pressure),
        float(shot.one_on_one),
        float(shot.first_time),
    ]


def shot_categorical_features(shot: Shot) -> dict[str, str]:
    return {
        "body_part": shot.body_part,
        "technique": shot.technique,
        "shot_type": shot.shot_type,
    }