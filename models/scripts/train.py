"""Train either the transferable xG model or the PIVOT state-value model."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from pivot.data import load_all_statsbomb_shots, split_shots_chronologically
from pivot.model import StateValueModel
from pivot.state import StateDataset
from pivot.xg import XGDataset, XGModel


DATA_DIR = MODELS_DIR / "data"
XG_CHECKPOINT = DATA_DIR / "processed" / "xg" / "best_model.pt"
STATE_CHECKPOINT = DATA_DIR / "processed" / "state_value" / "full_model.pt"


def _evaluate(model, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            inputs = [value.to(device) for value in batch[:-1]]
            targets = batch[-1].to(device)
            predictions = model(*inputs)
            loss = loss_fn(predictions, targets)
            count = targets.shape[0]
            total_loss += loss.item() * count
            total_examples += count
    return total_loss / total_examples


def _fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    checkpoint: Path,
    epochs: int,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            inputs = [value.to(device) for value in batch[:-1]]
            targets = batch[-1].to(device)
            optimizer.zero_grad()
            predictions = model(*inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            count = targets.shape[0]
            total_loss += loss.item() * count
            total_examples += count

        train_loss = total_loss / total_examples
        validation_loss = _evaluate(model, validation_loader, loss_fn, device)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint)
        print(
            f"Epoch {epoch + 1:02d} | train={train_loss:.6f} | "
            f"val={validation_loss:.6f}"
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return best_validation_loss, _evaluate(model, test_loader, loss_fn, device)


def train_xg(epochs: int = 40) -> None:
    torch.manual_seed(42)
    root = DATA_DIR / "raw" / "statsbomb"
    shots = load_all_statsbomb_shots(root)
    train_shots, validation_shots, test_shots = split_shots_chronologically(shots, root)
    loaders = [
        DataLoader(XGDataset(partition), batch_size=32, shuffle=index == 0)
        for index, partition in enumerate((train_shots, validation_shots, test_shots))
    ]
    best_loss, test_loss = _fit(
        XGModel(),
        *loaders,
        nn.BCEWithLogitsLoss(),
        XG_CHECKPOINT,
        epochs,
        torch.device("cpu"),
    )
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("Saved model:", XG_CHECKPOINT)


def _complete_dfl_matches(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / filename).exists() for filename in ("match.xml", "events.xml", "positions.xml"))
    )


def _split_dfl_matches(match_dirs: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    if len(match_dirs) < 3:
        raise RuntimeError("Need at least 3 complete DFL matches.")
    random.shuffle(match_dirs)
    train_count = max(1, int(len(match_dirs) * 0.6))
    validation_count = max(1, int(len(match_dirs) * 0.2))
    if train_count + validation_count >= len(match_dirs):
        train_count = len(match_dirs) - 2
        validation_count = 1
    return (
        match_dirs[:train_count],
        match_dirs[train_count : train_count + validation_count],
        match_dirs[train_count + validation_count :],
    )


def train_pivot(epochs: int = 25) -> None:
    torch.manual_seed(42)
    random.seed(42)
    partitions = _split_dfl_matches(_complete_dfl_matches(DATA_DIR / "raw" / "dfl"))
    labels = ("Train", "Validation", "Test")
    for label, paths in zip(labels, partitions):
        print(f"{label}:", [path.name for path in paths])

    datasets = [
        StateDataset(
            match_dirs=paths,
            xg_checkpoint=XG_CHECKPOINT,
            tracking_hz=5.0,
            sample_hz=1.0,
            future_window=15.0,
        )
        for paths in partitions
    ]
    for label, dataset in zip(labels, datasets):
        print(f"{label} samples:", len(dataset))

    loaders = [
        DataLoader(dataset, batch_size=256, shuffle=index == 0)
        for index, dataset in enumerate(datasets)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # MSE fits the conditional mean of future xG differential.
    best_loss, test_loss = _fit(
        StateValueModel(),
        *loaders,
        nn.MSELoss(),
        STATE_CHECKPOINT,
        epochs,
        device,
    )
    targets = torch.tensor([sample[-1].item() for sample in datasets[0]])
    zero_baseline = torch.mean(targets**2).item()
    print(f"Best validation MSE: {best_loss:.6f}")
    print(f"Test MSE: {test_loss:.6f}")
    print(f"Zero baseline MSE: {zero_baseline:.6f}")
    print("Saved model:", STATE_CHECKPOINT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="model", required=True)
    xg_parser = subparsers.add_parser("xg", help="train the transferable shot xG model")
    xg_parser.add_argument("--epochs", type=int, default=40)
    pivot_parser = subparsers.add_parser("pivot", help="train the tracking state-value model")
    pivot_parser.add_argument("--epochs", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "xg":
        train_xg(args.epochs)
    else:
        train_pivot(args.epochs)


if __name__ == "__main__":
    main()

