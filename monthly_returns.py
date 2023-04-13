import os
import pandas as pd
from tqdm import tqdm


def calculate_monthly_returns():
    path = os.path.join("data", "security_prices")
    files = os.listdir(path)
    dfs = []

    for file in tqdm(files):
        df = pd.read_csv(os.path.join(path, file), usecols=["Date", "Adj Close"])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        monthly_returns = df.resample('M').last().pct_change().reset_index().dropna()
        monthly_returns["investment_month"] = monthly_returns["Date"].dt.to_period('M').astype(str)
        monthly_returns["ticker"] = file[:-4]
        monthly_returns.rename(columns={"Adj Close": "ret"}, inplace=True)
        monthly_returns = monthly_returns[["ticker", "investment_month", "ret"]]
        dfs.append(monthly_returns)

    pd.concat(dfs).to_csv(os.path.join("data", "monthly_returns.csv"), index=False)


def monthly_turnovers():
    path = os.path.join("data", "security_prices")
    files = os.listdir(path)
    dfs = []

    for file in tqdm(files):
        df = pd.read_csv(os.path.join(path, file), usecols=["Date", "Volume", "Close"])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        monthly_returns = df.resample('M').mean().reset_index().dropna()
        monthly_returns["investment_month"] = monthly_returns["Date"].dt.to_period('M').astype(str)
        monthly_returns["ticker"] = file[:-4]
        monthly_returns["Turnover"] = monthly_returns["Volume"] * monthly_returns["Close"]
        monthly_returns = monthly_returns[["ticker", "investment_month", "Volume", "Turnover"]]
        dfs.append(monthly_returns)

    pd.concat(dfs).to_csv(os.path.join("data", "monthly_volumes.csv"), index=False)


if __name__ == '__main__':
    calculate_monthly_returns()
    monthly_turnovers()