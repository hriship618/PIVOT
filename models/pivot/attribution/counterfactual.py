import torch

from pivot.models.state_value import StateValueModel


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
    """
    Estimate a focal player's marginal spatial value.

    Real state:
        V(S)

    Counterfactual:
        Replace player i with the average state of
        the other 10 teammates.

    Returns:
        V(S) - V(S_counterfactual)
    """

    real_value = predict_value(
        model,
        focal_players,
        opponent_players,
        ball,
        context,
    )

    counterfactual_players = (
        focal_players.clone()
    )

    teammate_mask = torch.ones(
        focal_players.shape[0],
        dtype=torch.bool,
    )

    teammate_mask[player_index] = False

    replacement = (
        focal_players[
            teammate_mask
        ].mean(dim=0)
    )

    counterfactual_players[
        player_index
    ] = replacement

    counterfactual_value = predict_value(
        model,
        counterfactual_players,
        opponent_players,
        ball,
        context,
    )

    return (
        real_value
        - counterfactual_value
    )