import torch

from pivot.models.shot_xg import ShotXGModel


def main():
    model = ShotXGModel(input_size=26)

    fake_batch = torch.randn(8, 26)

    logits = model(fake_batch)

    probabilities = torch.sigmoid(logits)

    print("Logit shape:", logits.shape)
    print("Probabilities:", probabilities)


if __name__ == "__main__":
    main()