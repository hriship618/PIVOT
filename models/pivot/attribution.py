"""Player attribution for PIVOT expected-goal-differential value.

PIVOT attributes changes in expected goal differential rather than assigning
fixed values to football actions.

Ordinary actions:
    action xGD = V(after) - V(before)

Restarts:
    action xGD = V(restart state) - V(pre-event state)

Shots:
    shot decision = xG - V(pre-shot state)
    finishing      = goal - xG

Spatial value is handled separately through player counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pivot.accounting import (
    ActionEvent,
    ActionType,
    finishing_value,
    restart_award_value,
    shot_decision_value,
    state_transition_value,
)
from pivot.data import FrameState
from pivot.model import StateValueModel
from pivot.state import encode_sequence


DEFAULT_TRACKING_HZ = 5.0
DEFAULT_HISTORY_SECONDS = 2.0

# Ordinary actions resolve quickly.
DEFAULT_EVENT_OFFSET_SECONDS = 0.4

# Restarts may take longer than ordinary open-play actions to settle.
RESTART_SEARCH_SECONDS = 6.0

# Red-card effects can take several seconds to appear in tracking.
CARD_SEARCH_SECONDS = 20.0

RESTART_TYPES = {
    ActionType.FREE_KICK_WON,
    ActionType.CORNER_WON,
    ActionType.PENALTY_WON,
}

CARD_TYPES = {
    ActionType.YELLOW_CARD,
    ActionType.RED_CARD,
}


@dataclass(frozen=True)
class AttributedAction:
    """xGD assigned to one normalized football event."""

    event: ActionEvent
    action_xgd: float
    outcome_xgd: float = 0.0

    @property
    def total_xgd(self) -> float:
        return (
            self.action_xgd
            + self.outcome_xgd
        )


def predict_value(
    model: StateValueModel,
    focal_players: torch.Tensor,
    opponent_players: torch.Tensor,
    ball: torch.Tensor,
    context: torch.Tensor,
) -> float:
    """Evaluate exactly one PIVOT state or temporal history."""

    # Attribution operates on one example at a time, so make the batch
    # dimension explicit.
    focal_players = focal_players.unsqueeze(0)
    opponent_players = opponent_players.unsqueeze(0)
    ball = ball.unsqueeze(0)
    context = context.unsqueeze(0)

    with torch.no_grad():
        value = model(
            focal_players,
            opponent_players,
            ball,
            context,
        )

    return value.squeeze().item()


def player_counterfactual_value(
    model: StateValueModel,
    focal_players: torch.Tensor,
    opponent_players: torch.Tensor,
    ball: torch.Tensor,
    context: torch.Tensor,
    player_index: int,
) -> float:
    """Estimate spatial value contributed by one player's trajectory.

    The player's trajectory is replaced with the average trajectory of the
    other ten teammates.

    This remains the prototype spatial baseline. A learned/context-conditioned
    replacement can later replace this without changing the rest of PIVOT.
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

    num_players = (
        focal_players.shape[-2]
    )

    if not 0 <= player_index < num_players:
        raise IndexError(
            f"player_index {player_index} "
            f"outside [0, {num_players})"
        )

    teammate_mask = torch.ones(
        num_players,
        dtype=torch.bool,
        device=focal_players.device,
    )

    teammate_mask[player_index] = False

    # Works for both:
    #   [11, 4]
    # and
    #   [T, 11, 4]
    replacement = focal_players[
        ...,
        teammate_mask,
        :,
    ].mean(
        dim=-2
    )

    counterfactual_players[
        ...,
        player_index,
        :,
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


def _history_ending_at(
    frames: list[FrameState],
    period: int,
    timestamp: float,
    tracking_hz: float,
    history_seconds: float,
) -> list[FrameState] | None:
    """Return a valid tracking history ending at or before timestamp."""

    history_frames = max(
        1,
        round(
            tracking_hz
            * history_seconds
        ),
    )

    eligible = [
        frame
        for frame in frames
        if (
            frame.period == period
            and frame.timestamp <= timestamp
        )
    ]

    if len(eligible) < history_frames:
        return None

    history = eligible[
        -history_frames:
    ]

    current_period = history[-1].period

    if any(
        frame.period != current_period
        for frame in history
    ):
        return None

    # 20-22 tracked players allows legitimate red-card states while rejecting
    # obviously incomplete tracking.
    if any(
        len(frame.players) < 20
        for frame in history
    ):
        return None

    maximum_gap = (
        (1.0 / tracking_hz)
        * 1.75
    )

    for previous, current in zip(
        history,
        history[1:],
    ):
        gap = (
            current.timestamp
            - previous.timestamp
        )

        if (
            gap <= 0
            or gap > maximum_gap
        ):
            return None

    return history


def _state_value_at(
    model: StateValueModel,
    frames: list[FrameState],
    focal_team_id: str,
    attacking_right: bool,
    period: int,
    timestamp: float,
    tracking_hz: float,
    history_seconds: float,
) -> float | None:
    """Evaluate PIVOT using history ending near a requested timestamp."""

    history = _history_ending_at(
        frames,
        period,
        timestamp,
        tracking_hz,
        history_seconds,
    )

    if history is None:
        return None

    encoded = encode_sequence(
        history,
        focal_team_id,
        attacking_right,
    )

    return predict_value(
        model,
        *encoded,
    )


def _first_frame_matching(
    frames: list[FrameState],
    period: int,
    start_time: float,
    end_time: float,
    predicate,
) -> FrameState | None:
    """Find the earliest post-event frame satisfying a state condition."""

    for frame in frames:
        if frame.period != period:
            continue

        if frame.timestamp < start_time:
            continue

        if frame.timestamp > end_time:
            break

        if predicate(frame):
            return frame

    return None


def _restart_value(
    model: StateValueModel,
    event: ActionEvent,
    frames: list[FrameState],
    attacking_right: bool,
    tracking_hz: float,
    history_seconds: float,
) -> float | None:
    """Find and value the dead-ball state created by a restart award."""

    restart_frame = _first_frame_matching(
        frames=frames,
        period=event.period,
        start_time=event.timestamp,
        end_time=(
            event.timestamp
            + RESTART_SEARCH_SECONDS
        ),
        predicate=lambda frame: (
            not frame.ball_in_play
        ),
    )

    if restart_frame is None:
        return None

    return _state_value_at(
        model=model,
        frames=frames,
        focal_team_id=event.team_id,
        attacking_right=attacking_right,
        period=event.period,
        timestamp=restart_frame.timestamp,
        tracking_hz=tracking_hz,
        history_seconds=history_seconds,
    )


def _red_card_value(
    model: StateValueModel,
    event: ActionEvent,
    frames: list[FrameState],
    attacking_right: bool,
    tracking_hz: float,
    history_seconds: float,
) -> float | None:
    """Find the first tracked short-handed state after a dismissal."""

    short_handed_frame = _first_frame_matching(
        frames=frames,
        period=event.period,
        start_time=event.timestamp,
        end_time=(
            event.timestamp
            + CARD_SEARCH_SECONDS
        ),
        predicate=lambda frame: (
            len(frame.players) < 22
        ),
    )

    if short_handed_frame is None:
        return None

    return _state_value_at(
        model=model,
        frames=frames,
        focal_team_id=event.team_id,
        attacking_right=attacking_right,
        period=event.period,
        timestamp=short_handed_frame.timestamp,
        tracking_hz=tracking_hz,
        history_seconds=history_seconds,
    )


def attribute_action_event(
    model: StateValueModel,
    event: ActionEvent,
    frames: list[FrameState],
    attacking_right: bool,
    tracking_hz: float = DEFAULT_TRACKING_HZ,
    history_seconds: float = DEFAULT_HISTORY_SECONDS,
    event_offset_seconds: float = DEFAULT_EVENT_OFFSET_SECONDS,
) -> AttributedAction | None:
    """Estimate expected-GD contribution caused by one football event."""

    if (
        event.team_id is None
        or event.player_id is None
    ):
        return None

    # ------------------------------------------------------------------
    # STATE BEFORE EVENT
    # ------------------------------------------------------------------

    before_value = _state_value_at(
        model=model,
        frames=frames,
        focal_team_id=event.team_id,
        attacking_right=attacking_right,
        period=event.period,
        timestamp=(
            event.timestamp
            - event_offset_seconds
        ),
        tracking_hz=tracking_hz,
        history_seconds=history_seconds,
    )

    if before_value is None:
        return None

    # ------------------------------------------------------------------
    # SHOTS
    # ------------------------------------------------------------------

    if event.event_type == ActionType.SHOT:
        if event.xg is None:
            return None

        # The shooter receives only the improvement from the pre-shot state
        # to the shot's expected outcome, not all value used to create it.
        action_xgd = shot_decision_value(
            pre_shot_value=before_value,
            xg=event.xg,
        )

        outcome_xgd = 0.0

        if event.goal is not None:
            outcome_xgd = finishing_value(
                goal=event.goal,
                xg=event.xg,
            )

        return AttributedAction(
            event=event,
            action_xgd=action_xgd,
            outcome_xgd=outcome_xgd,
        )

    # ------------------------------------------------------------------
    # FREE KICKS / CORNERS / PENALTIES WON
    # ------------------------------------------------------------------

    if event.event_type in RESTART_TYPES:
        restart_value = _restart_value(
            model=model,
            event=event,
            frames=frames,
            attacking_right=attacking_right,
            tracking_hz=tracking_hz,
            history_seconds=history_seconds,
        )

        if restart_value is None:
            return None

        # Example:
        #   pre-foul state = +0.04
        #   resulting FK   = +0.17
        #
        #   player created +0.13 xGD by winning the restart.
        action_xgd = restart_award_value(
            pre_event_value=before_value,
            restart_value=restart_value,
        )

        return AttributedAction(
            event=event,
            action_xgd=action_xgd,
        )

    # ------------------------------------------------------------------
    # RED CARDS
    # ------------------------------------------------------------------

    if event.event_type == ActionType.RED_CARD:
        post_card_value = _red_card_value(
            model=model,
            event=event,
            frames=frames,
            attacking_right=attacking_right,
            tracking_hz=tracking_hz,
            history_seconds=history_seconds,
        )

        if post_card_value is None:
            return None

        # The event is evaluated from the carded team's perspective, so the
        # resulting contribution should naturally be strongly negative.
        action_xgd = state_transition_value(
            value_before=before_value,
            value_after=post_card_value,
        )

        return AttributedAction(
            event=event,
            action_xgd=action_xgd,
        )

    # ------------------------------------------------------------------
    # YELLOW CARDS
    # ------------------------------------------------------------------

    if event.event_type == ActionType.YELLOW_CARD:
        # The current state representation does not include accumulated player
        # disciplinary status, so inventing immediate value here would add
        # noise. This can later be learned once card status enters state input.
        return AttributedAction(
            event=event,
            action_xgd=0.0,
        )

    # ------------------------------------------------------------------
    # ORDINARY LIVE-BALL ACTIONS
    # ------------------------------------------------------------------

    after_value = _state_value_at(
        model=model,
        frames=frames,
        focal_team_id=event.team_id,
        attacking_right=attacking_right,
        period=event.period,
        timestamp=(
            event.timestamp
            + event_offset_seconds
        ),
        tracking_hz=tracking_hz,
        history_seconds=history_seconds,
    )

    if after_value is None:
        return None

    action_xgd = state_transition_value(
        value_before=before_value,
        value_after=after_value,
    )

    return AttributedAction(
        event=event,
        action_xgd=action_xgd,
    )


def _foul_is_followed_by_restart(
    event: ActionEvent,
    events: list[ActionEvent],
    tolerance_seconds: float = 1.5,
) -> bool:
    """Detect a restart representing the same incident as a foul.

    This prevents:

        foul state change
        +
        free-kick/penalty state change

    from both receiving credit for the same stoppage.
    """

    if event.event_type != ActionType.FOUL_CONCEDED:
        return False

    for other in events:
        if (
            other.period != event.period
            or other.event_type not in RESTART_TYPES
        ):
            continue

        if abs(
            other.timestamp
            - event.timestamp
        ) <= tolerance_seconds:
            return True

    return False


def attribute_actions(
    model: StateValueModel,
    events: list[ActionEvent],
    frames: list[FrameState],
    attacking_directions: dict[
        tuple[str, int],
        bool,
    ],
    tracking_hz: float = DEFAULT_TRACKING_HZ,
    history_seconds: float = DEFAULT_HISTORY_SECONDS,
    event_offset_seconds: float = DEFAULT_EVENT_OFFSET_SECONDS,
) -> list[AttributedAction]:
    """Attribute all usable events in one match."""

    attributed: list[
        AttributedAction
    ] = []

    for event in events:
        if event.team_id is None:
            continue

        # If a foul produced an explicitly represented restart, use the restart
        # event as the accounting boundary instead of counting both.
        if _foul_is_followed_by_restart(
            event,
            events,
        ):
            continue

        direction = (
            attacking_directions.get(
                (
                    event.team_id,
                    event.period,
                )
            )
        )

        if direction is None:
            continue

        credit = attribute_action_event(
            model=model,
            event=event,
            frames=frames,
            attacking_right=direction,
            tracking_hz=tracking_hz,
            history_seconds=history_seconds,
            event_offset_seconds=event_offset_seconds,
        )

        if credit is not None:
            attributed.append(
                credit
            )

    return attributed