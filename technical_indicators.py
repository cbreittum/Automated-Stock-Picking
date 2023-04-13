import os
import random
import warnings
import pandas as pd
from tqdm import tqdm
import ta
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def create_indicators():
    path = os.path.join("data", "security_prices")
    files = os.listdir(path)
    random.shuffle(files)

    for file in tqdm(files):
        ticker = file[:-4]
        res_file = os.path.join("data", "technical_indicators", file)

        if not os.path.isfile(res_file):
            try:
                data = pd.read_csv(os.path.join(path, file), usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
                df = ta.add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
                df = df.drop(["Open", "High", "Low", "Volume"], axis=1)

                for col in df.columns:
                    if col not in ["Date", "Close"]:
                        df[col + "_rel"] = df[col] / df["Close"]

                df = df.drop("Close", axis=1)
                df["ticker"] = ticker
                df.to_csv(res_file, index=False)
            except:
                pass


def monthly_indicators():
    dates = [f"{year}-{month:02d}-01" for year in range(1990, 2023) for month in range(1, 13)]
    res_file = os.path.join("data", "monthly_technical_indicators.csv")
    files = os.listdir(os.path.join("data", "technical_indicators"))

    monthly_indicators = []

    for file in tqdm(files):
        file_dfs = []
        df = pd.read_csv(os.path.join("data", "technical_indicators", file))
        min_date, max_date = min(df["Date"]), max(df["Date"])
        max_date_plus_one_month = (datetime.strptime(max_date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")

        for date in dates:
            if min_date < date < max_date_plus_one_month:
                df_date = df[df["Date"] < date]

                if not df_date.empty:
                    df_last_date = df_date.iloc[-1:].copy()
                    df_last_date["investment_month"] = date[:7]
                    file_dfs.append(df_last_date)

        if file_dfs:
            file_df = pd.concat(file_dfs)
            cols = [col for col in file_df.columns if col not in ["ticker", "Date", "investment_month"]]
            file_df = file_df[["ticker", "Date", "investment_month"] + cols]
            monthly_indicators.append(file_df)

    pd.concat(monthly_indicators).to_csv(res_file, index=False)


if __name__ == '__main__':
    create_indicators()
    monthly_indicators()