import numpy as np


def pf_ret(weights: np.ndarray, mean: np.ndarray, rf: float) -> float:
    return weights @ mean + (1 - weights.sum()) * rf


def pf_std(weights: np.ndarray, cov: np.ndarray) -> float:
    return np.sqrt(weights @ cov @ weights)  # type: ignore -> OK
