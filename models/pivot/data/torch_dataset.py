import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pivot.data.schema import Shot
from pivot.features.encoding import encode_shot


class ShotDataset(Dataset):
    def __init__(self, shots: list[Shot]):
        self.features: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for shot in shots:
            features, target = encode_shot(shot)

            self.features.append(
                torch.tensor(features, dtype=torch.float32)
            )

            self.targets.append(
                torch.tensor(target, dtype=torch.float32)
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


def split_shots_chronologically(
    shots: list[Shot],
    statsbomb_root: str | Path,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
):
    statsbomb_root = Path(statsbomb_root)

    matches_path = statsbomb_root / "matches.json"

    with matches_path.open("r", encoding="utf-8") as file:
        matches = json.load(file)

    ordered_matches = sorted(
        matches,
        key=lambda match: (
            match["match_date"],
            match["kick_off"],
            match["match_id"],
        ),
    )

    match_ids = [
        int(match["match_id"])
        for match in ordered_matches
    ]

    num_matches = len(match_ids)

    train_end = int(num_matches * train_fraction)

    validation_end = int(
        num_matches
        * (train_fraction + validation_fraction)
    )

    train_ids = set(match_ids[:train_end])

    validation_ids = set(
        match_ids[train_end:validation_end]
    )

    test_ids = set(
        match_ids[validation_end:]
    )

    train_shots = [
        shot
        for shot in shots
        if shot.match_id in train_ids
    ]

    validation_shots = [
        shot
        for shot in shots
        if shot.match_id in validation_ids
    ]

    test_shots = [
        shot
        for shot in shots
        if shot.match_id in test_ids
    ]

    return train_shots, validation_shots, test_shots