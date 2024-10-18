import matplotlib.pyplot as plt
import numpy as np

from ..pf import pf_ret, pf_std
from ..annualize import annualize_ret, annualize_std


def risk_return_plot(
    weights: np.ndarray, mean: np.ndarray, cov: np.ndarray, rf: float
) -> None:
    # func constants
    n_random = 300
    n_risky = weights.size
    n_ret_range = 1000
    scale_ret_range = 2

    # user's portfolio performance
    _pf_ret = pf_ret(weights, mean, rf)
    _pf_std = pf_std(weights, cov)

    # efficient frontier computation
    inv_cov = np.linalg.inv(cov)
    adj_ret = mean - rf
    one = np.ones_like(weights)
    s11 = one @ inv_cov @ one
    s1m = one @ inv_cov @ mean
    smm = mean @ inv_cov @ mean
    d = smm * s11 - s1m**2
    ef_ret_range = np.linspace(0, scale_ret_range * _pf_ret, n_ret_range)
    lin_ef_std = np.abs(ef_ret_range - rf) / np.sqrt(adj_ret @ inv_cov @ adj_ret)
    hyp_ef_std = np.sqrt((s11 * ef_ret_range**2 - 2 * s1m * ef_ret_range + smm) / d)

    # generating some random portfolio performances
    random_weights = np.random.normal(
        loc=weights.mean(), scale=weights.std(), size=[n_random, n_risky]
    )
    random_ret = np.zeros(n_random)
    random_std = np.zeros(n_random)
    for i in range(n_random):
        random_ret[i] = pf_ret(random_weights[i], mean, rf)
        random_std[i] = pf_std(random_weights[i], cov)

    # plotting
    plt.scatter(
        100 * annualize_std(_pf_std),
        100 * annualize_ret(_pf_ret),
        label="current pf",
        color="red",
        zorder=4,
    )
    plt.scatter(
        100 * annualize_std(random_std),  # type: ignore -> OK
        100 * annualize_ret(random_ret),  # type: ignore -> OK
        color="gray",
        label=f"{n_random} random pf",
        zorder=2,
    )
    plt.plot(
        100 * annualize_std(lin_ef_std),  # type: ignore -> OK
        100 * annualize_ret(ef_ret_range),  # type: ignore -> OK
        color="orange",
        label="with rf",
        zorder=3,
    )
    plt.plot(
        100 * annualize_std(hyp_ef_std),  # type: ignore -> OK
        100 * annualize_ret(ef_ret_range),  # type: ignore -> OK
        color="blue",
        label="without rf",
        zorder=3,
    )

    # figure edition
    plt.xlabel("Volatility (%)")
    plt.ylabel("Return (%)")
    plt.title("Risk-return plot")
    plt.legend()
    plt.grid(zorder=1)
    plt.tight_layout()
