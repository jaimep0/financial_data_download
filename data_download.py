#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import json
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def get_financials(ticker):
    stock = yf.Ticker(ticker)

    income = stock.quarterly_income_stmt.T
    balance = stock.quarterly_balance_sheet.T
    cashflow = stock.quarterly_cashflow.T

    financials = income.join(balance, how="outer")
    financials = financials.join(
        cashflow,
        how="outer",
        rsuffix="_cashflow"
    )

    financials.index.name = "period_end"
    financials = financials.sort_index().reset_index()
    financials["ticker"] = ticker

    return financials


def get_company_data(ticker):
    stock = yf.Ticker(ticker)
    df = get_financials(ticker)

    if df.empty:
        return df

    df["period_end"] = pd.to_datetime(
        df["period_end"], utc=True
    ).dt.tz_localize(None).astype("datetime64[ns]")

    # Historical shares from balance sheet
    for col in ["Ordinary Shares Number", "Share Issued"]:
        if col in df.columns:
            df["sharesOutstanding"] = pd.to_numeric(
                df[col], errors="coerce"
            )
            break

    # Company metadata
    try:
        info = stock.info
        for field in ["sector", "industry", "country"]:
            df[field] = info.get(field)
    except Exception:
        for field in ["sector", "industry", "country"]:
            df[field] = None

    return df


def get_market_data(tickers, start, end, verbose=True):
    frames = []

    for ticker in tqdm(
        tickers,
        desc="Market data",
        disable=not verbose
    ):
        try:
            prices = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                progress=False
            )

            if prices.empty:
                continue

            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(0)

            prices = prices.reset_index()

            prices = prices.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close_raw",
                "Adj Close": "close",
                "Volume": "volume",
                "Dividends": "dividend",
                "Stock Splits": "stock_splits"
            })

            prices["ticker"] = ticker

            prices["date"] = pd.to_datetime(
                prices["date"], utc=True
            ).dt.tz_localize(None).astype("datetime64[ns]")

            frames.append(prices)

        except Exception as e:
            if verbose:
                tqdm.write(f"Failed {ticker}: {e}")

    if not frames:
        raise ValueError("No market data downloaded.")

    return pd.concat(frames, ignore_index=True)


def extract_data(tickers, verbose=True):
    symbols = list(dict.fromkeys(list(tickers)))
    financials = []

    # 1. Extract quarterly financial statements
    for ticker in tqdm(
        symbols,
        desc="Financial statements",
        disable=not verbose
    ):
        try:
            df = get_company_data(ticker)

            if not df.empty:
                financials.append(df)

        except Exception as e:
            if verbose:
                tqdm.write(f"Failed {ticker}: {e}")

    if not financials:
        raise ValueError("No financial statements downloaded.")

    df = pd.concat(financials, ignore_index=True)

    # 2. Download historical prices
    start = (
        df["period_end"].min() - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    end = min(
        df["period_end"].max() + pd.Timedelta(days=10),
        pd.Timestamp.now(tz="UTC").tz_localize(None)
    ) + pd.Timedelta(days=1)

    end = end.strftime("%Y-%m-%d")

    prices = get_market_data(
        df["ticker"].unique().tolist(),
        start,
        end,
        verbose
    )

    # 3. Ensure identical datetime precision
    df["period_end"] = pd.to_datetime(
        df["period_end"]
    ).astype("datetime64[ns]")

    prices["date"] = pd.to_datetime(
        prices["date"]
    ).astype("datetime64[ns]")

    # 4. Match each quarter to its historical price
    df = pd.merge_asof(
        df.sort_values("period_end"),
        prices.sort_values("date"),
        left_on="period_end",
        right_on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=7)
    )

    # 5. Convert financial values to numeric types
    exclude = {
        "ticker",
        "period_end",
        "date",
        "sector",
        "industry",
        "country"
    }

    numeric_cols = [
        col for col in df.columns
        if col not in exclude
    ]

    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float64")

    df = df.reset_index(drop=True)

    # 6. Save one CSV in the current directory
    df.to_csv("data.csv", index=False)

    if verbose:
        print("\nExtraction completed")
        print(f"Observations: {len(df):,}")
        print(f"Companies: {df['ticker'].nunique()}")
        print(f"Columns: {df.shape[1]}")
        print("Saved: data.csv")

    return df


# In[3]:

# USAGE
# Import tickers from json file 
with open("tickers.json", "r") as file:
    tickers = json.load(file)
    
tickers = pd.DataFrame(tickers["companies"])["ticker"]

# Run extract_data to get a dataframe of financial data
df = extract_data(tickers)


# In[4]:


print(df.shape)

# save df
df.to_csv("data.csv")

