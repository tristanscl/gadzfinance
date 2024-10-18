import numpy as np


def holdings_to_weights(holdings: np.ndarray, last: np.ndarray) -> np.ndarray:
    pf_value = holdings @ last
    return holdings * last / pf_value


def weights_to_holdings(
    weights: np.ndarray, last: np.ndarray, pf_value: float
) -> np.ndarray:
    return weights * pf_value / last
