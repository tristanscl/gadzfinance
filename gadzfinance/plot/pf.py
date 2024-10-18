import matplotlib.pyplot as plt
import numpy as np


def weights_bar(weights: np.ndarray, assets: list[str]) -> None:
    # copies
    weights = weights.copy()
    assets = assets.copy()

    # adding the rf asset
    weights = np.append(weights, 1 - weights.sum())
    assets.append("rf")

    # plotting
    plt.bar(assets, weights * 100, zorder=2)

    # figure edition
    plt.grid(zorder=1)
    plt.ylabel("%")
    plt.title("Asset allocation (negative means short, sum > 1 means leverage.)")
    plt.tight_layout()
