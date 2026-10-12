"""PyTorch state-value architecture used by PIVOT checkpoints and inference."""

import torch
from torch import nn


class PlayerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(self, players: torch.Tensor) -> torch.Tensor:
        return self.network(players)


class StateValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.player_encoder = PlayerEncoder()
        # 32 focal + 32 opponent + 4 ball + 3 match context = 71 features.
        self.state_network = nn.Sequential(
            nn.Linear(71, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        focal_players: torch.Tensor,
        opponent_players: torch.Tensor,
        ball: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Mean aggregation keeps the learned value invariant to player row order.
        focal_state = self.player_encoder(focal_players).mean(dim=-2)
        opponent_state = self.player_encoder(opponent_players).mean(dim=-2)
        combined = torch.cat([focal_state, opponent_state, ball, context], dim=-1)
        return self.state_network(combined).squeeze(-1)

