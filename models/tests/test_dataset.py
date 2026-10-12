from pivot.data.dataset import load_all_statsbomb_shots


def main():
    shots = load_all_statsbomb_shots(
        "models/data/raw/statsbomb"
    )

    goals = sum(shot.goal for shot in shots)

    print(f"Total shots: {len(shots)}")
    print(f"Total goals: {goals}")
    print(f"Goal rate: {goals / len(shots):.3f}")


if __name__ == "__main__":
    main()