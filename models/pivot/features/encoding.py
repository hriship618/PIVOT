#encoding for pytorch to ingest
from pivot.data.schema import Shot
from pivot.features.shot import shot_numeric_features


BODY_PARTS = [
    "Left Foot",
    "Right Foot",
    "Head",
    "Other",
    "Unknown",
]

TECHNIQUES = [
    "Normal",
    "Volley",
    "Half Volley",
    "Lob",
    "Backheel",
    "Overhead Kick",
    "Diving Header",
    "Unknown",
]

SHOT_TYPES = [
    "Open Play",
    "Free Kick",
    "Penalty",
    "Corner",
    "Kick Off",
    "Unknown",
]


def one_hot(value: str, categories: list[str]) -> list[float]:
    if value not in categories:
        value = "Unknown"

    return [
        1.0 if value == category else 0.0
        for category in categories
    ]


def encode_shot(shot: Shot) -> tuple[list[float], float]:
    features = shot_numeric_features(shot)

    features += one_hot(
        shot.body_part,
        BODY_PARTS,
    )

    features += one_hot(
        shot.technique,
        TECHNIQUES,
    )

    features += one_hot(
        shot.shot_type,
        SHOT_TYPES,
    )

    target = float(shot.goal)

    return features, target