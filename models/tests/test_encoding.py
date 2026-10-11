from pivot.data.dataset import load_all_statsbomb_shots
from pivot.features.encoding import encode_shot


def main():
    shots = load_all_statsbomb_shots(
        "models/data/raw/statsbomb"
    )

    features, target = encode_shot(shots[0])

    print("Feature count:", len(features))
    print("Features:", features)
    print("Target:", target)


if __name__ == "__main__":
    main()