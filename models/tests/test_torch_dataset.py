from pivot.data.dataset import load_all_statsbomb_shots
from pivot.data.torch_dataset import (
    ShotDataset,
    split_shots_by_match,
)


def main():
    shots = load_all_statsbomb_shots(
        "models/data/raw/statsbomb"
    )

    train_shots, val_shots, test_shots = split_shots_by_match(
        shots
    )

    train_dataset = ShotDataset(train_shots)
    val_dataset = ShotDataset(val_shots)
    test_dataset = ShotDataset(test_shots)

    print("Train shots:", len(train_dataset))
    print("Validation shots:", len(val_dataset))
    print("Test shots:", len(test_dataset))

    features, target = train_dataset[0]

    print()
    print("Feature tensor shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Target:", target)


if __name__ == "__main__":
    main()