"""Expected-goal-differential accounting for PIVOT.

PIVOT measures expected goal differential added, not value over replacement.

Every credited contribution belongs to one of three non-overlapping buckets:

1. action_xgd
   Value created or destroyed by explicit football actions.

2. spatial_xgd
   Value created through positioning and off-ball movement between actions.

3. outcome_xgd
   Difference between an expected outcome and the realized outcome, primarily
   finishing and goalkeeper shot stopping.

The separation is intentional: a contribution should never be counted in more
than one bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

class ActionType(StrEnum):
    PASS = "pass"
    CROSS = "cross"
    CARRY = "carry"
    DRIBBLE = "dribble"

    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    TURNOVER = "turnover"

    FOUL_WON = "foul_won"
    FOUL_CONCEDED = "foul_conceded"
    FREE_KICK_WON = "free_kick_won"
    CORNER_WON = "corner_won"

    PENALTY_WON = "penalty_won"
    PENALTY_CONCEDED = "penalty_conceded"

    SHOT = "shot"

    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"

    OTHER = "other"

@dataclass(frozen=True)
class ActionEvent:
    """Provider-neutral event consumed by PIVOT's accounting layer."""

    match_id: str
    period: int
    timestamp: float
    event_type: ActionType

    team_id: str | None
    player_id: str | None

    x: float | None = None
    y: float | None = None

    opponent_team_id: str | None = None
    opponent_player_id: str | None = None

    successful: bool | None = None

    xg: float | None = None
    goal: bool | None = None

    qualifier: str | None = None

    # Kept so events can later be deduplicated or linked to provider metadata.
    provider_event_id: str | None = None

@dataclass
class PlayerContribution:
    """Running xGD ledger for one player."""

    action_xgd: float = 0.0
    spatial_xgd: float = 0.0
    outcome_xgd: float = 0.0

    @property
    def total_xgd(self) -> float:
        return self.action_xgd + self.spatial_xgd + self.outcome_xgd


def state_transition_value(
    value_before: float,
    value_after: float,
) -> float:
    """Expected GD created by changing one game state into another.

    This is the fundamental value equation for ordinary football actions.

        action value = V(after) - V(before)

    A line-breaking pass, tackle, interception, turnover, carry, etc. therefore
    receives whatever value the action actually created in its context rather
    than a fixed hand-written weight.
    """
    return value_after - value_before


def shot_decision_value(
    pre_shot_value: float,
    xg: float,
) -> float:
    """Value of choosing the shot, excluding finishing.

    The chance itself already exists in the pre-shot state. Giving the shooter
    the entire xG would steal creation credit from passes, movement and other
    players who created the chance.

    We therefore value the decision to shoot relative to the value of the
    position immediately before the shot.
    """
    return xg - pre_shot_value


def finishing_value(
    goal: bool,
    xg: float,
) -> float:
    """Finishing above or below expectation."""
    return float(goal) - xg


def goalkeeper_shot_stopping_value(
    goal: bool,
    xg: float,
) -> float:
    """Goalkeeper value relative to the expected shot outcome."""
    return xg - float(goal)


def penalty_award_value(
    pre_event_value: float,
    penalty_xg: float,
) -> float:
    """Value created when a player wins a penalty.

    The penalty taker does not receive this creation value. The player who
    caused the penalty to be awarded receives it; the taker's contribution is
    handled by shot/finishing accounting.
    """
    return penalty_xg - pre_event_value


def restart_award_value(
    pre_event_value: float,
    restart_value: float,
) -> float:
    """Value of winning a free kick or corner.

    `restart_value` is learned from data as the expected xGD of the resulting
    restart state. It is deliberately not a fixed lookup table.
    """
    return restart_value - pre_event_value


def card_value(
    pre_event_value: float,
    post_event_value: float,
) -> float:
    """Value of a sending-off through the resulting manpower state."""
    return post_event_value - pre_event_value


def add_action(
    ledger: dict[str, PlayerContribution],
    player_id: str | None,
    value: float,
) -> None:
    if player_id is None:
        return
    ledger.setdefault(player_id, PlayerContribution()).action_xgd += value


def add_spatial(
    ledger: dict[str, PlayerContribution],
    player_id: str | None,
    value: float,
) -> None:
    if player_id is None:
        return
    ledger.setdefault(player_id, PlayerContribution()).spatial_xgd += value


def add_outcome(
    ledger: dict[str, PlayerContribution],
    player_id: str | None,
    value: float,
) -> None:
    if player_id is None:
        return
    ledger.setdefault(player_id, PlayerContribution()).outcome_xgd += value


def per_90(value: float, minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    return 90.0 * value / minutes