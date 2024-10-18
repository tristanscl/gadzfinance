import matplotlib.pyplot as plt
import numpy as np

from ..annualize import annualize_ret, annualize_std


def mean_bar(mean: np.ndarray, assets: list[str], rf: float) -> None:
    # copies
    mean = mean.copy()
    assets = assets.copy()

    # adding the risk free asset
    mean = np.append(mean, rf)
    assets.append("rf")

    # plotting
    plt.bar(assets, annualize_ret(mean) * 100, zorder=2)  # type: ignore -> vectorized

    # figure editing
    plt.grid(zorder=1)
    plt.ylabel("%")
    plt.title("Annual mean returns")
    plt.tight_layout()


def std_heatmap(cov, assets) -> None:
    # creating a figure
    fig, ax = plt.subplots()

    # processing
    std = annualize_std((np.linalg.cholesky(cov))) * 100
    std = np.where(std == 0, np.nan, std).T

    # plotting
    cax = ax.matshow(std, cmap="coolwarm")
    cbar = fig.colorbar(cax)

    # figure editing
    ax.set_xticks(np.arange(len(assets)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(assets)
    ax.set_yticklabels(assets)
    cbar.set_label("%")
    ax.set_title("Annualized std")
    fig.tight_layout()
