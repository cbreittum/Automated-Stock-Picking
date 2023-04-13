import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def calculate_performance_metrics(path, perc):
    results = []

    for file in tqdm(os.listdir(path)):
        date = file[:-4]
        df = pd.read_csv(os.path.join(path, file))
        long_portf, short_portf = df.iloc[:int(len(df) / perc)], df.iloc[-int(len(df) / perc):]
        long_short_df = pd.concat([long_portf, short_portf])
        acc = accuracy_score(long_short_df["true_return_class"], long_short_df["predicted_class"])
        long_ew, short_ew = np.mean(long_portf["ret"]), np.mean(short_portf["ret"])
        ls_ret_mean = long_ew - short_ew
        ls_ret_median = np.median(df["ret"].iloc[:int(len(df) / perc)]) - np.median(
            df["ret"].iloc[-int(len(df) / perc):])

        results.append({"date": date, "Accuracy": acc, "long_ret": long_ew, "short_ret": short_ew, "ls_ret_mean": ls_ret_mean, "ls_ret_median": ls_ret_median})

    return pd.DataFrame(results).set_index("date")


if __name__ == '__main__':
    perc = 20
    path = "results//ML_results"
    total_df = calculate_performance_metrics(path, perc)

    print(total_df.to_string())
    print(total_df.mean())