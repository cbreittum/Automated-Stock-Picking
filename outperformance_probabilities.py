import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.mstats import winsorize


def load_and_preprocess_data(train_test_boundary):
    technical_indicators = pd.read_csv(os.path.join("data", "monthly_technical_indicators.csv"))

    # Drop low liquidity stocks
    monthly_volumes = pd.read_csv(os.path.join("data", "monthly_volumes.csv"))
    monthly_volumes = monthly_volumes[monthly_volumes["Turnover"] >= 1000000]
    technical_indicators = pd.merge(technical_indicators, monthly_volumes[["ticker", "investment_month"]],
                                    on=["ticker", "investment_month"], how="inner")

    monthly_returns = pd.read_csv(os.path.join("data", "monthly_returns.csv"))

    monthly_returns["ret"] = winsorize(monthly_returns["ret"], limits=(0, 0.01))
    technical_indicators = pd.merge(technical_indicators, monthly_returns, how="inner", on=["ticker", "investment_month"])
    monthly_medians = technical_indicators[["investment_month", "ret"]].groupby("investment_month").median()
    monthly_medians.rename(columns={"ret": "median_ret"}, inplace=True)
    technical_indicators = pd.merge(technical_indicators, monthly_medians, how="inner", on="investment_month")
    technical_indicators["true_return_class"] = [1 if value > technical_indicators["median_ret"].iloc[i] else 0 for i, value in enumerate(technical_indicators["ret"])]
    technical_indicators = technical_indicators.replace([np.inf, -np.inf], np.nan).dropna()

    for col in tqdm(technical_indicators.columns):
        if col not in ["Date", "investment_month", "ticker", "median_ret", "ret"]:
            technical_indicators = technical_indicators[(technical_indicators[col] > -np.finfo(np.float32).max) & (technical_indicators[col] < np.finfo(np.float32).max)]

    technical_indicators_train = technical_indicators[technical_indicators["Date"] < train_test_boundary].drop(["Date", "investment_month", "ticker", "median_ret", "ret"], axis=1)
    technical_indicators_test = technical_indicators[technical_indicators["Date"] >= train_test_boundary]

    return technical_indicators_train, technical_indicators_test


def train_and_save_model(technical_indicators_train):
    labels = list(technical_indicators_train["true_return_class"])
    technical_indicators_train = technical_indicators_train.drop(['true_return_class'], axis=1)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)
    model = rf.fit(technical_indicators_train, labels)
    pickle.dump(model, open(os.path.join('models', 'rf.save'), 'wb'))


def predict_and_save_results(investment_months, model, technical_indicators_test):
    for date in tqdm(investment_months):
        print("Predicting Date " + str(date))
        investment_df = technical_indicators_test[technical_indicators_test["investment_month"] == date]
        predicted_probas = model.predict_proba(investment_df.drop(['true_return_class', 'ret', 'ticker', 'Date', 'investment_month', 'median_ret'], axis=1))
        predicted_probas_class_1 = [value[1] for value in predicted_probas]
        result_df = investment_df[["ticker", 'Date', 'investment_month', 'ret', 'true_return_class']]
        result_df["probability"] = predicted_probas_class_1
        result_df = result_df.sort_values("probability", ascending=False)
        cutoff_proba = 0.5
        predicted_values = [1 if value > cutoff_proba else 0 for value in list(result_df["probability"])]
        result_df["predicted_class"] = predicted_values
        result_df.to_csv(os.path.join("results", "ML_results", "{}.csv".format(date)), index=False)

if __name__ == '__main__':
    train_test_boundary = "2002-12-31"
    investment_months = [f"{year}-{month:02d}" for year in range(2003, 2021) for month in range(1, 13)]
    technical_indicators_train, technical_indicators_test = load_and_preprocess_data(train_test_boundary)
    train_and_save_model(technical_indicators_train)
    model = pickle.load(open(os.path.join('models', 'rf.save'), 'rb'))
    predict_and_save_results(investment_months, model, technical_indicators_test)

