import torch

from pivot.data.dataset import load_all_statsbomb_shots
from pivot.data.torch_dataset import split_shots_chronologically
from pivot.features.encoding import encode_shot
from pivot.models.shot_xg import ShotXGModel


def main():
    root = "models/data/raw/statsbomb"

    shots = load_all_statsbomb_shots(root)

    _, _, test_shots = split_shots_chronologically(
        shots,
        root,
    )

    model = ShotXGModel(input_size=26)

    model.load_state_dict(
        torch.load(
            "models/data/processed/shot_xg/best_model.pt"
        )
    )

    model.eval()

    predictions = []

    with torch.no_grad():
        for shot in test_shots:
            features, _ = encode_shot(shot)

            x = torch.tensor(
                features,
                dtype=torch.float32,
            )

            logit = model(x)

            probability = torch.sigmoid(logit).item()

            predictions.append(
                (probability, shot)
            )

    predictions.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    print("Highest predicted xG shots")
    print("---------------------------")

    for probability, shot in predictions[:20]:
        print(
            f"xG={probability:.3f} | "
            f"goal={shot.goal} | "
            f"x={shot.x:.1f}, y={shot.y:.1f} | "
            f"{shot.body_part} | "
            f"{shot.technique} | "
            f"{shot.shot_type} | "
            f"1v1={shot.one_on_one} | "
            f"pressure={shot.under_pressure}"
        )

    print()
    print("Penalties")
    print("---------")

    penalties = [
        (probability, shot)
        for probability, shot in predictions
        if shot.shot_type == "Penalty"
    ]

    for probability, shot in penalties:
        print(
            f"xG={probability:.3f} | "
            f"goal={shot.goal}"
        )


if __name__ == "__main__":
    main()