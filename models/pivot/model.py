"""Spatiotemporal PIVOT model for expected goal-differential valuation.

Each frame is encoded with relation-aware attention over all 22 players.
A GRU models how those spatial states evolve over a short history.

Pairwise spatial relations use a custom CUDA kernel when available and fall
back to a pure PyTorch implementation otherwise.
"""

from __future__ import annotations

import math

import torch
from torch import nn


try:
    import pivot_cuda
except ImportError:
    pivot_cuda = None


PLAYER_FEATURES = 4
BALL_FEATURES = 4
CONTEXT_FEATURES = 5

NUM_PLAYERS_PER_TEAM = 11
NUM_PLAYERS = 22

MODEL_DIM = 96
NUM_HEADS = 4
NUM_SPATIAL_LAYERS = 3
FF_DIM = 192

RELATION_FEATURES = 6

BALL_DIM = 32
CONTEXT_DIM = 16

FRAME_DIM = 128
TEMPORAL_DIM = 128
TEMPORAL_LAYERS = 2


def pairwise_relations_torch(
    players: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of spatial player relations.

    Input:
        players [B, N, 4]

    Output:
        relations [B, N, N, 6]

    Each directed relation i -> j contains:

        dx
        dy
        distance
        relative_vx
        relative_vy
        closing_speed
    """

    positions = players[..., :, :2]
    velocities = players[..., :, 2:4]

    # relation[i, j] describes player j relative to player i.
    delta_position = (
        positions.unsqueeze(-3)
        - positions.unsqueeze(-2)
    )

    delta_velocity = (
        velocities.unsqueeze(-3)
        - velocities.unsqueeze(-2)
    )

    dx = delta_position[..., 0]
    dy = delta_position[..., 1]

    distance = torch.sqrt(
        dx.square()
        + dy.square()
        + 1e-8
    )

    dvx = delta_velocity[..., 0]
    dvy = delta_velocity[..., 1]

    unit_dx = dx / distance
    unit_dy = dy / distance

    # Positive closing speed means the players are approaching each other.
    closing_speed = -(
        dvx * unit_dx
        + dvy * unit_dy
    )

    return torch.stack(
        [
            dx,
            dy,
            distance,
            dvx,
            dvy,
            closing_speed,
        ],
        dim=-1,
    )


def pairwise_relations(
    players: torch.Tensor,
) -> torch.Tensor:
    """Compute spatial relations using CUDA when possible.

    The rest of PIVOT calls this function without caring whether the custom
    CUDA extension or the reference PyTorch implementation is being used.
    """

    if (
        pivot_cuda is not None
        and players.is_cuda
        and players.dtype == torch.float32
    ):
        return pivot_cuda.spatial_relations(
            players.contiguous()
        )

    return pairwise_relations_torch(
        players
    )


class RelationBias(nn.Module):
    """Convert pairwise spatial relations into attention-head biases."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                RELATION_FEATURES,
                32,
            ),
            nn.GELU(),
            nn.Linear(
                32,
                NUM_HEADS,
            ),
        )

    def forward(
        self,
        relations: torch.Tensor,
    ) -> torch.Tensor:
        # [B, N, N, H] -> [B, H, N, N]
        return self.network(
            relations
        ).movedim(
            -1,
            1,
        )


class RelationAttention(nn.Module):
    """Multi-head self-attention with learned spatial relation biases."""

    def __init__(self) -> None:
        super().__init__()

        if MODEL_DIM % NUM_HEADS != 0:
            raise ValueError(
                "MODEL_DIM must be divisible by NUM_HEADS"
            )

        self.head_dim = (
            MODEL_DIM // NUM_HEADS
        )

        self.qkv = nn.Linear(
            MODEL_DIM,
            3 * MODEL_DIM,
        )

        self.output = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.relation_bias = (
            RelationBias()
        )

    def forward(
        self,
        tokens: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, num_players, _ = (
            tokens.shape
        )

        qkv = self.qkv(
            tokens
        ).reshape(
            batch_size,
            num_players,
            3,
            NUM_HEADS,
            self.head_dim,
        )

        query, key, value = (
            qkv.unbind(dim=2)
        )

        # [B, N, H, D] -> [B, H, N, D]
        query = query.transpose(
            1,
            2,
        )

        key = key.transpose(
            1,
            2,
        )

        value = value.transpose(
            1,
            2,
        )

        scores = torch.matmul(
            query,
            key.transpose(
                -2,
                -1,
            ),
        )

        scores = (
            scores
            / math.sqrt(
                self.head_dim
            )
        )

        scores = (
            scores
            + self.relation_bias(
                relations
            )
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        attended = torch.matmul(
            weights,
            value,
        )

        attended = (
            attended
            .transpose(1, 2)
            .reshape(
                batch_size,
                num_players,
                MODEL_DIM,
            )
        )

        return self.output(
            attended
        )


class SpatialBlock(nn.Module):
    """One relation-aware Transformer block."""

    def __init__(self) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            MODEL_DIM
        )

        self.attention = (
            RelationAttention()
        )

        self.norm2 = nn.LayerNorm(
            MODEL_DIM
        )

        self.feed_forward = (
            nn.Sequential(
                nn.Linear(
                    MODEL_DIM,
                    FF_DIM,
                ),
                nn.GELU(),
                nn.Linear(
                    FF_DIM,
                    MODEL_DIM,
                ),
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        relations: torch.Tensor,
    ) -> torch.Tensor:

        tokens = (
            tokens
            + self.attention(
                self.norm1(tokens),
                relations,
            )
        )

        tokens = (
            tokens
            + self.feed_forward(
                self.norm2(tokens)
            )
        )

        return tokens


class StateValueModel(nn.Module):
    """Predict future expected xGD from a short tracking sequence."""

    def __init__(self) -> None:
        super().__init__()

        # Four movement features plus two team-identity flags.
        self.player_embedding = (
            nn.Sequential(
                nn.Linear(
                    PLAYER_FEATURES + 2,
                    MODEL_DIM,
                ),
                nn.GELU(),
                nn.Linear(
                    MODEL_DIM,
                    MODEL_DIM,
                ),
            )
        )

        self.spatial_blocks = (
            nn.ModuleList(
                SpatialBlock()
                for _ in range(
                    NUM_SPATIAL_LAYERS
                )
            )
        )

        self.ball_encoder = (
            nn.Sequential(
                nn.Linear(
                    BALL_FEATURES,
                    BALL_DIM,
                ),
                nn.GELU(),
                nn.Linear(
                    BALL_DIM,
                    BALL_DIM,
                ),
            )
        )

        self.context_encoder = (
            nn.Sequential(
                nn.Linear(
                    CONTEXT_FEATURES,
                    CONTEXT_DIM,
                ),
                nn.GELU(),
                nn.Linear(
                    CONTEXT_DIM,
                    CONTEXT_DIM,
                ),
            )
        )

        frame_input_dim = (
            MODEL_DIM * 2
            + BALL_DIM
            + CONTEXT_DIM
        )

        self.frame_encoder = (
            nn.Sequential(
                nn.Linear(
                    frame_input_dim,
                    FRAME_DIM,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    FRAME_DIM
                ),
            )
        )

        # A short temporal model captures runs, presses, recovery movement,
        # defensive line changes, and other dynamics invisible in one frame.
        self.temporal_encoder = nn.GRU(
            input_size=FRAME_DIM,
            hidden_size=TEMPORAL_DIM,
            num_layers=TEMPORAL_LAYERS,
            batch_first=True,
        )

        self.value_head = (
            nn.Sequential(
                nn.Linear(
                    TEMPORAL_DIM,
                    64,
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    1,
                ),
            )
        )

    def _encode_spatial_frames(
        self,
        focal_players: torch.Tensor,
        opponent_players: torch.Tensor,
        ball: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Encode every timestep using the same spatial interaction network."""

        (
            batch_size,
            sequence_length,
            _,
            _,
        ) = focal_players.shape

        players = torch.cat(
            [
                focal_players,
                opponent_players,
            ],
            dim=2,
        )

        # Treat every frame as an independent batch element while running the
        # spatial network. Temporal structure is restored afterward.
        players_flat = players.reshape(
            batch_size
            * sequence_length,
            NUM_PLAYERS,
            PLAYER_FEATURES,
        )

        focal_flag = torch.cat(
            [
                torch.ones(
                    NUM_PLAYERS_PER_TEAM,
                    1,
                    device=players.device,
                    dtype=players.dtype,
                ),
                torch.zeros(
                    NUM_PLAYERS_PER_TEAM,
                    1,
                    device=players.device,
                    dtype=players.dtype,
                ),
            ],
            dim=0,
        )

        opponent_flag = (
            1.0 - focal_flag
        )

        focal_flag = (
            focal_flag
            .unsqueeze(0)
            .expand(
                batch_size
                * sequence_length,
                -1,
                -1,
            )
        )

        opponent_flag = (
            opponent_flag
            .unsqueeze(0)
            .expand(
                batch_size
                * sequence_length,
                -1,
                -1,
            )
        )

        player_features = torch.cat(
            [
                players_flat,
                focal_flag,
                opponent_flag,
            ],
            dim=-1,
        )

        tokens = self.player_embedding(
            player_features
        )

        # This is the operation accelerated by the custom CUDA extension.
        relations = pairwise_relations(
            players_flat
        )

        for block in self.spatial_blocks:
            tokens = block(
                tokens,
                relations,
            )

        focal_tokens = tokens[
            :,
            :NUM_PLAYERS_PER_TEAM,
        ]

        opponent_tokens = tokens[
            :,
            NUM_PLAYERS_PER_TEAM:,
        ]

        # Pool only after player interactions have been modeled.
        focal_summary = (
            focal_tokens.mean(
                dim=1
            )
        )

        opponent_summary = (
            opponent_tokens.mean(
                dim=1
            )
        )

        ball_flat = ball.reshape(
            batch_size
            * sequence_length,
            BALL_FEATURES,
        )

        context_flat = (
            context.reshape(
                batch_size
                * sequence_length,
                CONTEXT_FEATURES,
            )
        )

        ball_summary = (
            self.ball_encoder(
                ball_flat
            )
        )

        context_summary = (
            self.context_encoder(
                context_flat
            )
        )

        frame_features = torch.cat(
            [
                focal_summary,
                opponent_summary,
                ball_summary,
                context_summary,
            ],
            dim=-1,
        )

        frame_embeddings = (
            self.frame_encoder(
                frame_features
            )
        )

        return frame_embeddings.reshape(
            batch_size,
            sequence_length,
            FRAME_DIM,
        )

    def forward(
        self,
        focal_players: torch.Tensor,
        opponent_players: torch.Tensor,
        ball: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate single frames, batches, or temporal sequences.

        Supported player shapes:

            [11, 4]
            [B, 11, 4]
            [B, T, 11, 4]
        """

        scalar_input = (
            focal_players.ndim == 2
        )

        # Single unbatched frame.
        if focal_players.ndim == 2:
            focal_players = (
                focal_players
                .unsqueeze(0)
                .unsqueeze(0)
            )

            opponent_players = (
                opponent_players
                .unsqueeze(0)
                .unsqueeze(0)
            )

            ball = (
                ball
                .unsqueeze(0)
                .unsqueeze(0)
            )

            context = (
                context
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Batch of independent frames.
        elif focal_players.ndim == 3:
            focal_players = (
                focal_players
                .unsqueeze(1)
            )

            opponent_players = (
                opponent_players
                .unsqueeze(1)
            )

            ball = ball.unsqueeze(1)
            context = (
                context.unsqueeze(1)
            )

        elif focal_players.ndim != 4:
            raise ValueError(
                "Expected player tensor shaped "
                "[11,4], [B,11,4], or [B,T,11,4]"
            )

        if (
            focal_players.shape[-2]
            != NUM_PLAYERS_PER_TEAM
            or opponent_players.shape[-2]
            != NUM_PLAYERS_PER_TEAM
        ):
            raise ValueError(
                "Expected 11 encoded slots per team"
            )

        frames = (
            self._encode_spatial_frames(
                focal_players,
                opponent_players,
                ball,
                context,
            )
        )

        temporal_output, _ = (
            self.temporal_encoder(
                frames
            )
        )

        # The final GRU state summarizes the complete history through time t.
        current_state = (
            temporal_output[
                :,
                -1,
                :,
            ]
        )

        value = (
            self.value_head(
                current_state
            )
            .squeeze(-1)
        )

        if scalar_input:
            return value.squeeze(0)

        return value