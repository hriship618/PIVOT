# PIVOT model pipeline

Run these commands from this `models/` directory:

```bash
python scripts/train.py xg
python scripts/train.py pivot
python scripts/build_rankings.py
```

The xG command trains the five-feature portable model against StatsBomb xG labels and writes `data/processed/xg/best_model.pt`. The pivot command trains the DFL state-value model and writes `data/processed/state_value/full_model.pt`. The ranking command loads that checkpoint and writes frontend-compatible JSON and CSV under `data/processed/rankings/`.

```text
models/
├── pivot/
│   ├── data.py          # StatsBomb and DFL schemas/parsers
│   ├── xg.py            # xG features, dataset, model, inference
│   ├── state.py         # orientation, encoding, targets, state dataset
│   ├── model.py         # state-value network
│   └── attribution.py   # player counterfactual value
├── scripts/
│   ├── train.py
│   └── build_rankings.py
├── tests/
│   └── test_pipeline.py
└── data/
    ├── raw/{statsbomb,dfl}/
    └── processed/{xg,state_value,rankings}/
```

Both teams are rotated into the focal team's left-to-right attacking direction before encoding. The model predicts the next-15-second xG differential. Attribution replaces one focal player with the mean state of the other ten teammates and reports the change in predicted value.

The ranking output preserves the frontend fields `rank`, `player_id`, `name`, `team_id`, `matches`, `minutes_sampled`, `spatial_value`, and `prototype_pivot`.

Validation does not require retraining:

```bash
python -m pytest -q
python -m compileall -q pivot scripts tests
```
