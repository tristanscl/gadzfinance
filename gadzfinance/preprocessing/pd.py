import pandas as pd
from typing import Tuple


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    return data / data.dropna().iloc[-1]


def get_returns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    return data / data.shift(1) - 1
