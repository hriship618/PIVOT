import torch
from torch.utils.data import DataLoader

from pivot.data.dataset import load_all_statsbomb_shots
from pivot.data.torch_dataset import (
    ShotDataset,
    split_shots_chronologically,
)
from pivot.models.shot_xg import ShotXGModel


def collect_predictions(
    model: ShotXGModel,
    loader: DataLoader,
):
    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():
        for features, batch_targets in loader:
            logits = model(features)

            batch_probabilities = torch.sigmoid(logits)

            probabilities.extend(
                batch_probabilities.tolist()
            )

            targets.extend(
                batch_targets.tolist()
            )

    return probabilities, targets


def brier_score(
    probabilities: list[float],
    targets: list[float],
) -> float:
    total = 0.0

    for probability, target in zip(
        probabilities,
        targets,
    ):
        total += (probability - target) ** 2

    return total / len(targets)


def log_loss(
    probabilities: list[float],
    targets: list[float],
) -> float:
    epsilon = 1e-7
    total = 0.0

    for probability, target in zip(
        probabilities,
        targets,
    ):
        probability = min(
            max(probability, epsilon),
            1 - epsilon,
        )

        total += -(
            target * torch.log(
                torch.tensor(probability)
            ).item()
            +
            (1 - target)
            * torch.log(
                torch.tensor(1 - probability)
            ).item()
        )

    return total / len(targets)


def auc(
    probabilities: list[float],
    targets: list[float],
) -> float:
    pairs = sorted(
        zip(probabilities, targets),
        key=lambda item: item[0],
    )

    positives = sum(targets)
    negatives = len(targets) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    rank_sum = 0.0

    for rank, (_, target) in enumerate(
        pairs,
        start=1,
    ):
        if target == 1:
            rank_sum += rank

    return (
        rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)


def calibration_table(
    probabilities: list[float],
    targets: list[float],
    bins: int = 10,
):
    rows = []

    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins

        bucket = [
            (p, y)
            for p, y in zip(
                probabilities,
                targets,
            )
            if (
                lower <= p < upper
                or (
                    bin_index == bins - 1
                    and p == 1.0
                )
            )
        ]

        if not bucket:
            continue

        mean_prediction = sum(
            p for p, _ in bucket
        ) / len(bucket)

        mean_target = sum(
            y for _, y in bucket
        ) / len(bucket)

        rows.append(
            (
                lower,
                upper,
                len(bucket),
                mean_prediction,
                mean_target,
            )
        )

    return rows


def main():
    root = "models/data/raw/statsbomb"

    shots = load_all_statsbomb_shots(root)

    train_shots,_, test_shots = split_shots_chronologically(
    shots,
    root,
)
    train_goal_rate = (
    sum(float(shot.goal) for shot in train_shots)
    / len(train_shots)
    )

    test_dataset = ShotDataset(test_shots)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    model = ShotXGModel(input_size=26)

    model.load_state_dict(
        torch.load(
            "models/data/processed/shot_xg/best_model.pt"
        )
    )

    probabilities, targets = collect_predictions(
        model,
        test_loader,
    )

    mean_prediction = sum(probabilities) / len(probabilities)
    actual_goal_rate = sum(targets) / len(targets)

    baseline_probability = train_goal_rate  

    baseline_predictions = [
        baseline_probability
        for _ in targets
    ]

    print("Model")
    print("-----")
    print(
        f"Mean predicted xG: {mean_prediction:.4f}"
    )
    print(
        f"Actual goal rate:  {actual_goal_rate:.4f}"
    )
    print(
        f"Brier score:       "
        f"{brier_score(probabilities, targets):.4f}"
    )
    print(
        f"Log loss:          "
        f"{log_loss(probabilities, targets):.4f}"
    )
    print(
        f"ROC AUC:           "
        f"{auc(probabilities, targets):.4f}"
    )

    print()
    print()
    print("Constant baseline")
    print("-----------------")
    print(
        f"Training goal rate: {train_goal_rate:.4f}"
    )
    print(
        f"Log loss:          "
        f"{log_loss(baseline_predictions, targets):.4f}"
    )

    print()
    print("Calibration")
    print("-----------")

    for (
        lower,
        upper,
        count,
        predicted,
        observed,
    ) in calibration_table(
        probabilities,
        targets,
    ):
        print(
            f"{lower:.1f}-{upper:.1f} | "
            f"n={count:3d} | "
            f"pred={predicted:.3f} | "
            f"actual={observed:.3f}"
        )


if __name__ == "__main__":
    main()