import json
import pandas as pd
import yfinance as yf


def get_stock_data(code, start, end):
    data = yf.download(code, start, end)
    data = data.drop(data[data.Volume < 10].index)
    business_date = pd.bdate_range(data.index[0], data.index[-1])
    data = pd.DataFrame(data, index=business_date)
    data.index.name = "Date"
    data["Adj Close"] = data["Adj Close"].interpolate(method="linear")
    return data


def stock_download(
    dic,
    start="2001-01-01",
    end="2021-11-30",
    len_data=5000,
    n_stock=50,
    download_dir="../data/stocks/",
):
    count = 0
    stock_dict = {}
    for symbol in dic:
        symbol = symbol if symbol != "BRK.B" else "BRK-B"
        data = get_stock_data(symbol, start, end)
        if len(data) > len_data:
            data.to_csv(download_dir + f"{symbol}.csv")
            stock_dict[symbol] = dic[symbol]
            count += 1
            print(symbol)
        else:
            print(f"failed at {symbol}")
        if count >= n_stock:
            break
    return stock_dict


if __name__ == "__main__":
    config = json.load(open("../config/data_config.json", "r", encoding="utf8"))
    snp500 = pd.read_csv("../data/snp500.csv")
    snp500.loc[snp500.Symbol == "BRK.B", "Symbol"] = "BRK-B"
    snp500 = {tup[2]: tup[1] for tup in snp500.values.tolist()}
    stock_pair = stock_download(
        snp500, len_data=config["LEN_DATA"], n_stock=config["N_STOCK"], download_dir='../data/stocks/'
    )
    sp500 = yf.download("^GSPC", config["START"], config["END"])
    sp500.to_csv("../data/snp500_index.csv")
    json.dump(stock_pair, open("../data/stock.json", "w", encoding="UTF-8"))
