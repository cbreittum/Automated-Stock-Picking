import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf

def download_stock_data(tickers, start_date, end_date):
    for ticker in tqdm(tickers):
        res_file = os.path.join("data", "security_prices", f"{ticker}.csv")
        if not os.path.isfile(res_file):
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(res_file)

if __name__ == '__main__':
    start_date, end_date = "1980-01-01", "2023-01-01"
    ticker_df = pd.read_csv(os.path.join("data", "ticker_cik.txt"), sep="\t")
    tickers = ticker_df["ticker"].str.upper().tolist()
    download_stock_data(tickers, start_date, end_date)