import torch
from torch import nn


class BallContextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        focal_players,
        opponent_players,
        ball,
        context,
    ):
        x = torch.cat(
            [ball, context],
            dim=-1,
        )

        return self.network(x).squeeze(-1)