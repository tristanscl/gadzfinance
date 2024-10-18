import yfinance as yf
import datetime as dt
import pandas as pd


def get_raw(assets: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    return yf.Tickers(assets).history(start=start, end=end)["Close"]


def clean(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    # copies
    data = data.copy()

    # index transtyping (datetime -> date)
    data.index = data.index.date  # type: ignore -> OK

    start = data.index[0]
    end = data.index[-1]

    # reindexing -> all dates (extend)
    all_dates = pd.date_range(start=start, end=end, freq="D").date
    data = data.reindex(all_dates)

    # ffill
    data = data.ffill()

    # reindexing -> clean dates (contract)
    clean_dates = pd.date_range(start=start, end=end, freq="B").date
    data = data.reindex(clean_dates)

    return data


def convert(data: pd.DataFrame, currency: str) -> pd.DataFrame:
    # copies
    data = data.copy()

    start = data.index[0]
    end = data.index[-1]

    for asset in data.columns:
        # getting asset currency
        asset_currency = yf.Ticker(asset).info["currency"]
        print(f"currency for asset {asset} is {asset_currency}")

        # changing asset currency
        if currency != asset_currency:
            change = yf.Ticker(currency + asset_currency + "=X").history(
                start=start, end=end
            )["Close"]
            change = clean(change)
            data[asset] /= change

    return data


def pipeline(
    assets: list[str], start: dt.date, end: dt.date, currency: str
) -> pd.DataFrame:
    data = get_raw(assets, start, end)
    data = clean(data)
    data = convert(data, currency)  # type: ignore -> OK
    return data
