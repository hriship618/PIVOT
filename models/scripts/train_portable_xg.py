from pathlib import Path

import torch

from torch import nn
from torch.utils.data import DataLoader

from pivot.data.dataset import (
    load_all_statsbomb_shots,
)

from pivot.data.torch_dataset import (
    split_shots_chronologically,
)

from pivot.data.portable_xg_dataset import (
    PortableXGDataset,
)

from pivot.models.portable_xg import (
    PortableXGModel,
)


def evaluate(
    model,
    loader,
    loss_fn,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for features, targets in loader:

            logits = model(features)

            loss = loss_fn(
                logits,
                targets,
            )

            batch_size = (
                features.shape[0]
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


def main():

    torch.manual_seed(42)

    root = (
        "models/data/raw/statsbomb"
    )

    shots = (
        load_all_statsbomb_shots(
            root
        )
    )

    (
        train_shots,
        val_shots,
        test_shots,
    ) = split_shots_chronologically(
        shots,
        root,
    )

    train_dataset = (
        PortableXGDataset(
            train_shots
        )
    )

    val_dataset = (
        PortableXGDataset(
            val_shots
        )
    )

    test_dataset = (
        PortableXGDataset(
            test_shots
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = PortableXGModel()

    # BCE works with soft targets,
    # so StatsBomb probabilities in [0,1]
    # are valid targets.
    loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 40

    checkpoint_path = Path(
        "models/data/processed/"
        "portable_xg/"
        "best_model.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_examples = 0

        for (
            features,
            targets,
        ) in train_loader:

            optimizer.zero_grad()

            logits = model(
                features
            )

            loss = loss_fn(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                features.shape[0]
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
                checkpoint_path,
            )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path
        )
    )

    test_loss = evaluate(
        model,
        test_loader,
        loss_fn,
    )

    print()
    print(
        "Best validation loss:",
        round(
            best_val_loss,
            4,
        ),
    )

    print(
        "Test loss:",
        round(
            test_loss,
            4,
        ),
    )


if __name__ == "__main__":
    main()