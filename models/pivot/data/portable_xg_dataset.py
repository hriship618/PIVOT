import torch

from torch.utils.data import Dataset

from pivot.data.schema import Shot
from pivot.features.portable_xg import (
    statsbomb_geometry_features,
)


class PortableXGDataset(Dataset):

    def __init__(
        self,
        shots: list[Shot],
    ):

        self.features = []
        self.targets = []

        for shot in shots:

            penalty = (
                shot.shot_type
                == "Penalty"
            )

            features = (
                statsbomb_geometry_features(
                    x=shot.x,
                    y=shot.y,
                    penalty=penalty,
                )
            )

            self.features.append(
                torch.tensor(
                    features,
                    dtype=torch.float32,
                )
            )

            self.targets.append(
                torch.tensor(
                    shot.statsbomb_xg,
                    dtype=torch.float32,
                )
            )

    def __len__(self):
        return len(
            self.features
        )

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.features[index],
            self.targets[index],
        )