from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerState:
    player_id: int
    team_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class BallState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class FrameState:
    match_id: int
    period: int
    timestamp: float

    home_team_id: int
    away_team_id: int
    possession_team_id: Optional[int]

    home_score: int
    away_score: int

    players: list[PlayerState]
    ball: BallState


@dataclass
class Shot:
    match_id: int
    player_id: int
    team_id: int

    period: int
    timestamp: float

    x: float
    y: float

    body_part: str
    technique: str
    shot_type: str

    under_pressure: bool
    one_on_one: bool
    first_time: bool

    goal: bool