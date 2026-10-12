from pathlib import Path

from pivot.data.statsbomb import load_shots
from pivot.features.shot import (
    shot_categorical_features,
    shot_numeric_features,
)


def main():
    path = Path(
        "models/data/raw/statsbomb/events/3857256.json"
    )

    shots = load_shots(path, match_id=3857256)

    for shot in shots[:5]:
        print()
        print(shot)

        print(
            "numeric:",
            shot_numeric_features(shot),
        )

        print(
            "categorical:",
            shot_categorical_features(shot),
        )


if __name__ == "__main__":
    main()