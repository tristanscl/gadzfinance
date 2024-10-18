import numpy as np
from .CONSTANTS import TRADING_YEAR


def deannualize_ret(ret: float) -> float:
    return ret / TRADING_YEAR


def deannualize_std(std: float) -> float:
    return std / np.sqrt(TRADING_YEAR)


def annualize_ret(ret: float) -> float:
    return ret * TRADING_YEAR


def annualize_std(std: float) -> float:
    return std * np.sqrt(TRADING_YEAR)
