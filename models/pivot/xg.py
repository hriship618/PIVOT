"""Portable shot-geometry xG features, dataset, model, and inference."""

from math import atan2, pi, sqrt
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from pivot.data import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    STATSBOMB_PITCH_LENGTH,
    STATSBOMB_PITCH_WIDTH,
    Shot,
)


GOAL_CENTER_Y = PITCH_WIDTH / 2
GOAL_WIDTH = 7.32


def shot_geometry_features(x_m: float, y_m: float, penalty: bool) -> list[float]:
    dx = PITCH_LENGTH - x_m
    dy = GOAL_CENTER_Y - y_m
    distance = sqrt(dx * dx + dy * dy)
    left_post_y = GOAL_CENTER_Y - GOAL_WIDTH / 2
    right_post_y = GOAL_CENTER_Y + GOAL_WIDTH / 2
    angle = abs(atan2(right_post_y - y_m, dx) - atan2(left_post_y - y_m, dx))
    return [
        x_m / PITCH_LENGTH,
        y_m / PITCH_WIDTH,
        distance / PITCH_LENGTH,
        angle / pi,
        float(penalty),
    ]


def statsbomb_geometry_features(x: float, y: float, penalty: bool) -> list[float]:
    # StatsBomb's 120x80 coordinates are mapped to the physical DFL pitch scale.
    return shot_geometry_features(
        x_m=x / STATSBOMB_PITCH_LENGTH * PITCH_LENGTH,
        y_m=y / STATSBOMB_PITCH_WIDTH * PITCH_WIDTH,
        penalty=penalty,
    )


class XGDataset(Dataset):
    def __init__(self, shots: list[Shot]):
        self.features = [
            torch.tensor(
                statsbomb_geometry_features(
                    shot.x,
                    shot.y,
                    penalty=shot.shot_type == "Penalty",
                ),
                dtype=torch.float32,
            )
            for shot in shots
        ]
        # The transferable model approximates StatsBomb xG, not raw goal outcomes.
        self.targets = [torch.tensor(shot.statsbomb_xg, dtype=torch.float32) for shot in shots]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class XGModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def load_xg_model(checkpoint_path: str | Path) -> XGModel:
    model = XGModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict_xg(model: XGModel, x: float, y: float, penalty: bool) -> float:
    features = torch.tensor(shot_geometry_features(x, y, penalty), dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(features)).item()

