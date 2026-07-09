"""Local, human-inspectable report rendering for the synthetic pipeline check."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(
    *, train_loss: list[float], validation_loss: list[float], output_path: Path
) -> None:
    """Plot train/validation loss vs. epoch and save it to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if train_loss:
        ax.plot(range(len(train_loss)), train_loss, label="train_loss", marker="o")
    if validation_loss:
        ax.plot(
            range(len(validation_loss)),
            validation_loss,
            label="validation_loss",
            marker="o",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Synthetic pipeline check: loss curve")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_path)
    finally:
        plt.close(fig)
