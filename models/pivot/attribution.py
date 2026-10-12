"""Counterfactual player attribution for a trained state-value model."""

import torch

from pivot.model import StateValueModel


def predict_value(
    model: StateValueModel,
    focal_players: torch.Tensor,
    opponent_players: torch.Tensor,
    ball: torch.Tensor,
    context: torch.Tensor,
) -> float:
    with torch.no_grad():
        value = model(
            focal_players.unsqueeze(0),
            opponent_players.unsqueeze(0),
            ball.unsqueeze(0),
            context.unsqueeze(0),
        )
    return value.item()


def player_counterfactual_value(
    model: StateValueModel,
    focal_players: torch.Tensor,
    opponent_players: torch.Tensor,
    ball: torch.Tensor,
    context: torch.Tensor,
    player_index: int,
) -> float:
    """Return V(observed) minus V(teammate-average replacement)."""
    real_value = predict_value(model, focal_players, opponent_players, ball, context)
    counterfactual_players = focal_players.clone()
    teammate_mask = torch.ones(focal_players.shape[0], dtype=torch.bool)
    teammate_mask[player_index] = False

    # Counterfactual attribution isolates the model's estimated spatial contribution
    # of a player relative to a neutral replacement state.
    counterfactual_players[player_index] = focal_players[teammate_mask].mean(dim=0)
    counterfactual_value = predict_value(
        model,
        counterfactual_players,
        opponent_players,
        ball,
        context,
    )
    return real_value - counterfactual_value

