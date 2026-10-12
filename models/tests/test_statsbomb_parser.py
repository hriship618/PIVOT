from pathlib import Path

from pivot.data.statsbomb import load_shots


def main():
    path = Path("models/data/raw/statsbomb/events/3857256.json")

    shots = load_shots(path, match_id=3857256)

    print(f"Loaded {len(shots)} shots")

    for shot in shots[:5]:
        print()
        print(shot)


if __name__ == "__main__":
    main()