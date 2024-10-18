import numpy as np
import pandas as pd
import datetime as dt
import copy
import matplotlib.pyplot as plt

from ..holdings import holdings_to_weights
from ..CONSTANTS import TRADING_YEAR


def forecast_plot(
    sim: np.ndarray,
    holdings: np.ndarray,
    last: np.ndarray,
    data: pd.DataFrame,
    rf: float,
    start: dt.date,
) -> None:
    # func constants
    n_levels = 10
    n_paths = 5
    opacity = 0.3
    epsilon = 0.01

    # copies
    sim = copy.deepcopy(sim)
    data = data.copy()

    # scaling the data
    weights = holdings_to_weights(holdings, last)
    data = data / data.loc[start]
    sim /= last

    # processing the data
    data_dates = data.index
    if not (-epsilon < 1 - weights.sum() < epsilon):  # risky only case
        data_rf_term = (1 - weights.sum()) * (
            np.ones(data_dates.size) * rf + 1
        ).cumprod()
        data_rf_term /= data_rf_term[-1]  # normalization
        data_perf = data.to_numpy() @ weights + data_rf_term
    else:
        data_perf = data.to_numpy() @ weights

    # processing the sim
    n_sim, n_days, n_assets = sim.shape
    sim_rf_term = (1 - weights.sum()) * (np.ones(n_days) * rf + 1).cumprod()
    sim_perf = sim @ weights + sim_rf_term
    new_dates = pd.date_range(start=start, periods=n_days, freq="B").date

    # plotting
    plt.plot(data_dates, data_perf, label="real perf", zorder=2)
    for q in np.linspace(1 / n_levels, 1 / 2, n_levels):
        lower = np.quantile(sim_perf, axis=0, q=q)
        upper = np.quantile(sim_perf, axis=0, q=1 - q)
        plt.fill_between(
            new_dates, lower, upper, alpha=2 / n_levels, color="gray", zorder=1
        )
    for i in range(n_paths):
        plt.plot(new_dates, sim_perf[i], alpha=opacity, zorder=1)

    # figure editing
    plt.legend()
    plt.grid(zorder=0)
    plt.xticks(rotation=45)
    plt.title(
        f"Foreast plot for the portfolio (GBM) for {int(n_days / TRADING_YEAR)} years"
    )
    plt.tight_layout()
