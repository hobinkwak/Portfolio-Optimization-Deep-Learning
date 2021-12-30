import json
import pickle
import numpy as np
import pandas as pd


def get_return_df(stock_dic, in_path="data/stocks/", out_path="data/"):
    for i, ticker in enumerate(stock_dic):
        stock = in_path + f"{ticker}.csv"
        stock_df = pd.read_csv(stock, index_col="Date")[["Adj Close"]]
        if i == 0:
            return_df = np.log(stock_df) - np.log(stock_df.shift(1))
            return_df.columns = [ticker]
        else:
            return_df[ticker] = np.log(stock_df) - np.log(stock_df.shift(1))
    return_df = return_df.dropna()
    return_df.to_csv(out_path + "return_df.csv")
    return return_df


def make_DL_dataset(data, data_len, n_stock):
    times = []
    dataset = np.array(data.iloc[:data_len, :]).reshape(1, -1, n_stock)
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i : data_len + i, :]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i : data_len + i, :].index)
    return dataset, times


def data_split(data, train_len, pred_len, tr_ratio, n_stock):
    return_train, times_train = make_DL_dataset(
        data[: int(len(data) * tr_ratio)], train_len + pred_len, n_stock
    )
    return_test, times_test = make_DL_dataset(
        data[int(len(data) * tr_ratio) :], train_len + pred_len, n_stock
    )

    x_tr = np.array([x[:train_len] for x in return_train])
    y_tr = np.array([x[-pred_len:] for x in return_train])
    times_tr = np.unique(
        np.array([x[-pred_len:] for x in times_train]).flatten()
    ).tolist()

    x_te = np.array([x[:train_len] for x in return_test])
    y_te = np.array([x[-pred_len:] for x in return_test])
    times_te = np.unique(
        np.array([x[-pred_len:] for x in times_test]).flatten()
    ).tolist()

    return x_tr, y_tr, x_te, y_te, times_tr, times_te


if __name__ == "__main__":
    path = "data/"
    config = json.load(open("config/data_config.json", "r", encoding="utf8"))
    stock_dict_sp = json.load(open(path + "stock.json", "r", encoding="UTF8"))
    return_df = get_return_df(stock_dict_sp)
    x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split(
        return_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        config["N_STOCK"],
    )

    with open(path + "date.pkl", "wb") as f:
        pickle.dump(times_te, f)

    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_te, y_te], f)
