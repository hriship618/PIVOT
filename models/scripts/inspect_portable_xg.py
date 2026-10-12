import torch

from pivot.data.dataset import load_all_statsbomb_shots
from pivot.data.torch_dataset import split_shots_chronologically
from pivot.features.portable_xg import statsbomb_geometry_features
from pivot.models.portable_xg import PortableXGModel


def main():
    root = "models/data/raw/statsbomb"

    shots = load_all_statsbomb_shots(root)

    _, _, test_shots = split_shots_chronologically(
        shots,
        root,
    )

    model = PortableXGModel()

    model.load_state_dict(
        torch.load(
            "models/data/processed/portable_xg/best_model.pt",
            map_location="cpu",
        )
    )

    model.eval()

    rows = []

    with torch.no_grad():
        for shot in test_shots:

            penalty = (
                shot.shot_type == "Penalty"
            )

            features = statsbomb_geometry_features(
                x=shot.x,
                y=shot.y,
                penalty=penalty,
            )

            x = torch.tensor(
                features,
                dtype=torch.float32,
            )

            predicted = torch.sigmoid(
                model(x)
            ).item()

            rows.append(
                (
                    predicted,
                    shot.statsbomb_xg,
                    shot,
                )
            )

    rows.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    print("Highest predicted xG shots")
    print("---------------------------")

    for predicted, actual_xg, shot in rows[:20]:
        print(
            f"pred={predicted:.3f} | "
            f"SB={actual_xg:.3f} | "
            f"x={shot.x:.1f}, y={shot.y:.1f} | "
            f"{shot.shot_type}"
        )

    print()
    print("Penalties")
    print("---------")

    for predicted, actual_xg, shot in rows:
        if shot.shot_type == "Penalty":
            print(
                f"pred={predicted:.3f} | "
                f"SB={actual_xg:.3f}"
            )


if __name__ == "__main__":
    main()