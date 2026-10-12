# PIVOT

**PIVOT** is a soccer player-value framework built to estimate how much a player changes expected goal differential through both on-ball actions and off-ball positioning.

The long-term goal is to build a soccer analogue of basketball metrics such as EPM: a single player-value estimate grounded in changes to expected scoring margin rather than raw box-score statistics.

This repository contains the current research/prototype implementation, including:

- event and tracking-data ingestion
- a portable expected-goals model
- spatiotemporal state-value modeling
- action-level expected goal differential attribution
- off-ball counterfactual attribution
- GPU training support
- a custom CUDA extension for pairwise spatial features
- a frontend for exploring prototype player rankings

> **Status:** research prototype. Current rankings are based on a small tracking-data sample and should not be interpreted as production-quality player evaluations.

---

## Motivation

Traditional soccer statistics capture only fragments of player value.

Goals and assists miss buildup and defending. Event-based expected-value models improve on this, but still struggle with actions away from the ball. Tracking data provides much richer information: player positioning, defensive shape, runs, spacing, pressure, and movement.

PIVOT models the game as a sequence of states and asks:

> **How does the expected goal differential change because of a player's actions and positioning?**

The current decomposition is:

\[
\text{PIVOT}_i =
\text{Action xGD}_i +
\text{Spatial xGD}_i +
\text{Outcome residual}_i
\]

where:

- **Action xGD** captures changes associated with passes, tackles, shots, restarts, and other events.
- **Spatial xGD** captures off-ball value using counterfactual player positioning.
- **Outcome residual** captures finishing or goalkeeping performance beyond expected outcomes.

Player contributions can then be standardized per 90 minutes.

---

## Model Overview

### 1. Expected Goals

PIVOT first trains a portable expected-goals model using StatsBomb event data.

The current model uses shot geometry features including:

- shot location
- distance to goal
- shot angle
- penalty indicator

The model is trained against StatsBomb xG values and is then used to assign a common xG scale to tracking datasets that do not provide the same shot-quality model.

---

### 2. State Representation

A match state contains:

- 11 focal-team players
- 11 opponent players
- ball position and velocity
- player position and velocity
- possession
- score differential
- match time
- ball-in-play state
- manpower differential

All states are normalized so that the focal team attacks from left to right.

Rather than modeling isolated frames, PIVOT uses short tracking sequences so the model can reason about motion and developing play.

---

### 3. Spatiotemporal State-Value Model

The state-value model estimates:

\[
V(S_t)
\approx
\mathbb{E}
\left[
\text{future xG for}
-
\text{future xG against}
\right]
\]

over a short future window.

The architecture combines:

#### Relation-aware player attention

For every ordered player pair \(i,j\), PIVOT computes:

\[
(\Delta x,\Delta y,d,\Delta v_x,\Delta v_y,c)
\]

where \(c\) represents closing speed.

These pairwise relations are passed into a relation network that produces attention biases for multi-head player interaction layers.

#### Team pooling

After spatial interaction, representations are pooled separately for:

- the focal team
- the opponent

This preserves the distinction between attacking and defending structure.

#### Ball and match context

Separate encoders process:

- ball state
- possession
- score
- time
- ball-in-play status
- manpower differential

#### Temporal modeling

Frame-level representations are passed through a multi-layer GRU to model how the state evolves over time.

The final temporal representation is mapped to predicted expected goal differential.

---

## Action Attribution

For ordinary actions, the current framework measures the local change in state value:

\[
\Delta V =
V(S_{\text{after}})
-
V(S_{\text{before}})
\]

This allows identical actions to receive different values depending on context.

For example, a tackle near one's own goal can be substantially more valuable than the same tackle in a harmless midfield state.

### Shots

Shot value is separated into two components.

#### Shot decision

\[
\text{Shot Decision Value}
=
xG -
V(S_{\text{pre-shot}})
\]

#### Finishing

\[
\text{Finishing Value}
=
\text{Goal} - xG
\]

This prevents the shooter from receiving all of the credit for chance creation while still rewarding finishing performance.

For goalkeepers, the corresponding shot-stopping residual is:

\[
xG - \text{Goal}
\]

### Restarts and cards

Special transitions such as:

- free kicks
- corners
- penalties
- red cards

are handled through explicit state transitions rather than being treated as ordinary open-play events.

---

## Off-Ball Attribution

Tracking data makes it possible to evaluate value that never appears in event logs.

For a focal player, PIVOT constructs a counterfactual state in which that player's trajectory is replaced by a baseline trajectory derived from teammates.

The difference between the original and counterfactual state values provides an estimate of the player's spatial contribution.

This is currently a prototype attribution method. More sophisticated role- and context-conditioned counterfactuals are a natural extension.

---

## CUDA Extension

PIVOT includes a custom CUDA kernel for computing pairwise player-relation features.

For a batch of player states with shape:

```text
[B, N, 4]