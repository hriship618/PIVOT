from pathlib import Path
import random

import torch

from torch import nn
from torch.utils.data import DataLoader

from pivot.data.state_dataset import (
    StateDataset,
)

from pivot.models.state_value import (
    StateValueModel,
)


def evaluate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for (
            focal,
            opponent,
            ball,
            context,
            target,
        ) in loader:

            focal = focal.to(device)
            opponent = opponent.to(device)
            ball = ball.to(device)
            context = context.to(device)
            target = target.to(device)

            prediction = model(
                focal,
                opponent,
                ball,
                context,
            )

            loss = loss_fn(
                prediction,
                target,
            )

            batch_size = (
                target.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_examples += (
                batch_size
            )

    return (
        total_loss
        / total_examples
    )


def target_stats(
    dataset,
):

    targets = torch.tensor(
        [
            dataset[i][-1].item()
            for i in range(
                len(dataset)
            )
        ]
    )

    return {
        "mean": targets.mean().item(),

        "std": targets.std().item(),

        "nonzero": (
            targets.abs() > 1e-8
        ).float().mean().item(),
    }


def main():

    torch.manual_seed(42)

    random.seed(42)

    dfl_root = Path(
        "models/data/raw/dfl"
    )

    match_dirs = [
        path
        for path
        in dfl_root.iterdir()

        if (
            path.is_dir()

            and (
                path / "match.xml"
            ).exists()

            and (
                path / "events.xml"
            ).exists()

            and (
                path / "positions.xml"
            ).exists()
        )
    ]

    match_dirs = sorted(
        match_dirs
    )

    if len(match_dirs) < 3:

        raise RuntimeError(
            "Need at least 3 complete "
            "DFL matches."
        )

    random.shuffle(
        match_dirs
    )

    num_matches = len(
        match_dirs
    )

    train_count = max(
        1,
        int(
            num_matches * 0.6
        ),
    )

    val_count = max(
        1,
        int(
            num_matches * 0.2
        ),
    )

    # Leave at least one match
    # entirely unseen for testing.
    if (
        train_count
        + val_count
        >= num_matches
    ):
        train_count = (
            num_matches - 2
        )

        val_count = 1

    train_dirs = (
        match_dirs[
            :train_count
        ]
    )

    val_dirs = (
        match_dirs[
            train_count:
            train_count + val_count
        ]
    )

    test_dirs = (
        match_dirs[
            train_count + val_count:
        ]
    )

    print()
    print("Match split")
    print("-----------")

    print(
        "Train:",
        [
            path.name
            for path
            in train_dirs
        ],
    )

    print(
        "Validation:",
        [
            path.name
            for path
            in val_dirs
        ],
    )

    print(
        "Test:",
        [
            path.name
            for path
            in test_dirs
        ],
    )

    checkpoint = (
        "models/data/processed/"
        "portable_xg/"
        "best_model.pt"
    )

    print()
    print("Building training dataset...")

    train_dataset = StateDataset(
        match_dirs=train_dirs,
        xg_checkpoint=checkpoint,

        tracking_hz=5.0,
        sample_hz=1.0,
        future_window=15.0,
    )

    print()
    print("Building validation dataset...")

    val_dataset = StateDataset(
        match_dirs=val_dirs,
        xg_checkpoint=checkpoint,

        tracking_hz=5.0,
        sample_hz=1.0,
        future_window=15.0,
    )

    print()
    print("Building test dataset...")

    test_dataset = StateDataset(
        match_dirs=test_dirs,
        xg_checkpoint=checkpoint,

        tracking_hz=5.0,
        sample_hz=1.0,
        future_window=15.0,
    )

    print()
    print("Dataset sizes")
    print("-------------")

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(val_dataset),
    )

    print(
        "Test:",
        len(test_dataset),
    )

    stats = target_stats(
        train_dataset
    )

    print()
    print("Training target")
    print("---------------")

    print(
        "Mean:",
        round(
            stats["mean"],
            5,
        ),
    )

    print(
        "Std:",
        round(
            stats["std"],
            5,
        ),
    )

    print(
        "Nonzero:",
        round(
            100
            * stats["nonzero"],
            2,
        ),
        "%",
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=256,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=256,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,

        batch_size=256,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    model = (
        StateValueModel()
        .to(device)
    )

    # We want the conditional mean
    # of future xG difference,
    # so squared error is appropriate.
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    model_checkpoint = Path(
        "models/data/processed/"
        "state_value/"
        "full_model.pt"
    )   

    model_checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    epochs = 25

    print()
    print("Training")
    print("--------")

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_examples = 0

        for (
            focal,
            opponent,
            ball,
            context,
            target,
        ) in train_loader:

            focal = focal.to(device)
            opponent = opponent.to(device)
            ball = ball.to(device)
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(
                focal,
                opponent,
                ball,
                context,
            )

            loss = loss_fn(
                prediction,
                target,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                target.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_examples += (
                batch_size
            )

        train_loss = (
            total_loss
            / total_examples
        )

        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            torch.save(
                model.state_dict(),
                model_checkpoint,
            )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

    model.load_state_dict(
        torch.load(
            model_checkpoint,
            map_location=device,
        )
    )

    test_loss = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    # Constant-zero baseline.
    zero_squared_error = 0.0
    zero_count = 0

    for sample in test_dataset:

        target = sample[-1].item()

        zero_squared_error += (
            target ** 2
        )

        zero_count += 1

    zero_baseline = (
        zero_squared_error
        / zero_count
    )

    print()
    print("Final results")
    print("-------------")

    print(
        "Best validation MSE:",
        round(
            best_val_loss,
            6,
        ),
    )

    print(
        "Test MSE:",
        round(
            test_loss,
            6,
        ),
    )

    print(
        "Zero baseline MSE:",
        round(
            zero_baseline,
            6,
        ),
    )

    print(
        "Saved model:",
        model_checkpoint,
    )


if __name__ == "__main__":
    main()