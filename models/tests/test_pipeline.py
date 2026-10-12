"""Focused invariants for the PIVOT data-to-ranking pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from pivot.attribution import player_counterfactual_value
from pivot.data import BallState, FrameState, PlayerState, load_tracking_frames
from pivot.model import StateValueModel
from pivot.state import encode_state, orient_frame, players_in_encoding_order
from pivot.xg import XGModel, predict_xg
from scripts.build_rankings import RANKING_FIELDS, export_rankings


MODELS_DIR = Path(__file__).resolve().parents[1]


def _frame() -> FrameState:
    players = []
    for index in range(11):
        players.append(
            PlayerState(f"home-{index}", "home", 10.0 + index, 5.0 + index, 1.0, -2.0)
        )
        players.append(
            PlayerState(f"away-{index}", "away", 70.0 + index, 40.0 + index, -3.0, 4.0)
        )
    return FrameState(
        match_id="match",
        period=1,
        timestamp=100.0,
        home_team_id="home",
        away_team_id="away",
        possession_team_id="home",
        home_score=2,
        away_score=1,
        ball_in_play=True,
        players=players,
        ball=BallState(70.0, 30.0, -6.0, 2.0),
    )


def _write_minimal_dfl_match(root: Path) -> None:
    teams = []
    for team_id, prefix in (("home", "h"), ("away", "a")):
        players = "".join(
            f'<Player PersonId="{prefix}{index}" Shortname="{prefix}{index}" />'
            for index in range(11)
        )
        teams.append(f'<Team TeamId="{team_id}"><Players>{players}</Players></Team>')
    (root / "match.xml").write_text(
        '<Root><MatchInformation><General MatchId="DFL-MAT-test" '
        'HomeTeamId="home" GuestTeamId="away" />'
        '<Environment PitchX="105" PitchY="68" />'
        f'<Teams>{"".join(teams)}</Teams></MatchInformation></Root>',
        encoding="utf-8",
    )
    (root / "events.xml").write_text(
        '<Root><Event EventTime="2024-01-01T12:00:00">'
        '<KickOff GameSection="firstHalf" TeamLeft="home" TeamRight="away" />'
        '</Event></Root>',
        encoding="utf-8",
    )
    frame_times = ("2024-01-01T12:00:00.000", "2024-01-01T12:00:00.040")
    ball_frames = "".join(
        f'<Frame N="{index}" T="{timestamp}" X="0" Y="0" '
        'BallPossession="1" BallStatus="1" />'
        for index, timestamp in enumerate(frame_times)
    )
    frame_sets = [f'<FrameSet TeamId="ball" GameSection="firstHalf">{ball_frames}</FrameSet>']
    for team_prefix in ("h", "a"):
        for player_index in range(11):
            player_frames = "".join(
                f'<Frame N="{index}" T="{timestamp}" X="{player_index}" Y="0" />'
                for index, timestamp in enumerate(frame_times)
            )
            frame_sets.append(
                f'<FrameSet PersonId="{team_prefix}{player_index}" '
                f'GameSection="firstHalf">{player_frames}</FrameSet>'
            )
    (root / "positions.xml").write_text(
        f'<Root>{"".join(frame_sets)}</Root>',
        encoding="utf-8",
    )


def test_dfl_live_frames_have_22_players(tmp_path: Path) -> None:
    _write_minimal_dfl_match(tmp_path)
    frames = load_tracking_frames(tmp_path, target_hz=25.0)
    live_frames = [frame for frame in frames if frame.ball_in_play]
    assert live_frames
    assert all(len(frame.players) == 22 for frame in live_frames)


def test_orientation_rotates_positions_and_velocities() -> None:
    original = _frame()
    rotated = orient_frame(original, attacking_right=False)
    assert (rotated.players[0].x, rotated.players[0].y) == (95.0, 63.0)
    assert (rotated.players[0].vx, rotated.players[0].vy) == (-1.0, 2.0)
    assert (rotated.ball.x, rotated.ball.y) == (35.0, 38.0)
    assert (rotated.ball.vx, rotated.ball.vy) == (6.0, -2.0)
    assert original.players[0].x == 10.0


def test_state_encoding_dimensions_and_player_order() -> None:
    frame = _frame()
    focal, opponents, ball, context = encode_state(frame, "home", True)
    ordered_players, _ = players_in_encoding_order(frame, "home", True)
    assert focal.shape == (11, 4)
    assert opponents.shape == (11, 4)
    assert ball.shape == (4,)
    assert context.shape == (3,)
    assert [player.player_id for player in ordered_players] == [f"home-{i}" for i in range(11)]
    assert torch.isclose(focal[3, 0], torch.tensor(ordered_players[3].x / 105.0))


def test_checkpoint_compatible_inference_outputs() -> None:
    xg_model = XGModel()
    xg_checkpoint = MODELS_DIR / "data" / "processed" / "xg" / "best_model.pt"
    if xg_checkpoint.exists():
        xg_model.load_state_dict(torch.load(xg_checkpoint, map_location="cpu"))
    probability = predict_xg(xg_model, x=90.0, y=34.0, penalty=False)
    assert 0.0 <= probability <= 1.0

    state_model = StateValueModel()
    state_checkpoint = MODELS_DIR / "data" / "processed" / "state_value" / "full_model.pt"
    if state_checkpoint.exists():
        state_model.load_state_dict(torch.load(state_checkpoint, map_location="cpu"))
    encoded = encode_state(_frame(), "home", True)
    assert state_model(*encoded).ndim == 0


def test_counterfactual_does_not_mutate_state() -> None:
    model = StateValueModel()
    focal, opponents, ball, context = encode_state(_frame(), "home", True)
    originals = [tensor.clone() for tensor in (focal, opponents, ball, context)]
    result = player_counterfactual_value(model, focal, opponents, ball, context, 3)
    assert isinstance(result, float)
    for tensor, original in zip((focal, opponents, ball, context), originals):
        assert torch.equal(tensor, original)


def test_rankings_export_frontend_schema(tmp_path: Path) -> None:
    ranking = {
        "player_id": "p1",
        "name": "Player One",
        "team_id": "team",
        "matches": 1,
        "minutes_sampled": 90.0,
        "spatial_value": 0.001,
        "prototype_pivot": 10.0,
        "rank": 1,
    }
    json_path, csv_path = export_rankings([ranking], tmp_path)
    assert set(json.loads(json_path.read_text(encoding="utf-8"))[0]) == set(RANKING_FIELDS)
    with csv_path.open(newline="", encoding="utf-8") as file:
        assert set(next(csv.DictReader(file))) == set(RANKING_FIELDS)

