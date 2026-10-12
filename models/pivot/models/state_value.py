import torch
from torch import nn


class PlayerEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(
        self,
        players: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(
            players
        )


class StateValueModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.player_encoder = (
            PlayerEncoder()
        )

        self.state_network = nn.Sequential(

            # focal team representation
            # opponent representation
            # ball
            # context
            #
            # 32 + 32 + 4 + 3 = 71

            nn.Linear(71, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        focal_players,
        opponent_players,
        ball,
        context,
    ):

        focal_embeddings = (
            self.player_encoder(
                focal_players
            )
        )

        opponent_embeddings = (
            self.player_encoder(
                opponent_players
            )
        )

        # DeepSets aggregation.
        #
        # It doesn't matter what order
        # the players are given in.

        focal_state = (
            focal_embeddings.mean(
                dim=-2
            )
        )

        opponent_state = (
            opponent_embeddings.mean(
                dim=-2
            )
        )

        combined = torch.cat(
            [
                focal_state,
                opponent_state,
                ball,
                context,
            ],
            dim=-1,
        )

        value = (
            self.state_network(
                combined
            )
        )

        return value.squeeze(-1)