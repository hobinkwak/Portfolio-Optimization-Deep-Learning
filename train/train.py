import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.gru import GRU
from model.transformer import TransAm
from model.sam import SAM
from model.loss import max_sharpe, equal_risk_parity
from train.utils import save_model, load_model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if self.config["USE_CUDA"] else "cpu"
        model_name = self.config["MODEL"]
        if model_name.lower() == "gru":
            self.model = GRU(
                self.config["N_LAYER"],
                self.config["HIDDEN_DIM"],
                self.config["N_FEAT"],
                self.config["DROPOUT"],
            ).to(self.device)
        if model_name.lower() == "transformer":
            self.model = TransAm(
                self.config["N_FEAT"],
                self.config["N_LAYER"],
                self.config["N_HEAD"],
                self.config["HIDDEN_DIM"],
                self.config["DROPOUT"],
            ).to(self.device)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(
            self.model.parameters(), base_optimizer, lr=self.config["LR"]
            , momentum=self.config['MOMENTUM']
        )
        self.criterion = max_sharpe


    def _dataload(self):
        with open("data/dataset.pkl", "rb") as f:
            train_x_raw, train_y_raw, test_x_raw, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            test_date = pickle.load(f)
        self.train_x_raw = train_x_raw
        self.train_y_raw = train_y_raw
        self.test_x_raw = test_x_raw
        self.test_y_raw = test_y_raw
        self.test_date = test_date

    def _scale_data(self, scale=21):
        self.train_x = torch.from_numpy(self.train_x_raw.astype("float32") * scale)
        self.train_y = torch.from_numpy(self.train_y_raw.astype("float32") * scale)
        self.test_x = torch.from_numpy(self.test_x_raw.astype("float32") * scale)
        self.test_y = torch.from_numpy(self.test_y_raw.astype("float32") * scale)

    def _set_parameter(self):
        self.LEN_TRAIN = self.train_x.shape[1]
        self.LEN_PRED = self.train_y.shape[1]
        self.N_STOCK = self.config["N_FEAT"]

    def _shuffle_data(self):
        randomized = np.arange(len(self.train_x))
        np.random.shuffle(randomized)
        self.train_x = self.train_x[randomized]
        self.train_y = self.train_y[randomized]

    def set_data(self):
        self._dataload()
        self._scale_data()
        self._set_parameter()
        self._shuffle_data()

    def dataloader(self, x, y):
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["BATCH"],
            shuffle=False,
            drop_last=True,
        )

    def train(self, visualize=True):
        train_loader = self.dataloader(self.train_x, self.train_y)
        test_loader = self.dataloader(self.test_x, self.test_y)

        valid_loss = []
        train_loss = []

        early_stop_count = 0
        early_stop_th = self.config["EARLY_STOP"]

        for epoch in range(self.config["EPOCHS"]):
            print("Epoch {}/{}".format(epoch + 1, self.config["EPOCHS"]))
            print("-" * 10)
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = test_loader

                running_loss = 0.0

                for idx, data in enumerate(dataloader):
                    x, y = data
                    x = x.to("cuda")
                    y = y.to("cuda")
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        out = self.model(x)
                        loss = self.criterion(y, out)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.first_step(zero_grad=True)
                            self.criterion(y, self.model(x)).backward()
                            self.optimizer.second_step(zero_grad=True)

                    running_loss += loss.item() / len(dataloader)
                if phase == "train":
                    train_loss.append(running_loss)
                else:
                    valid_loss.append(running_loss)
                    if running_loss <= min(valid_loss):
                        save_model(self.model, "result", "hb")
                        print(f"Improved! at {epoch + 1} epochs, with {running_loss}")
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stop_count == early_stop_th:
                break

        if visualize:
            self._visualize_training(train_loss, valid_loss)

        return self.model, train_loss, valid_loss

    def _visualize_training(self, train_loss, valid_loss):
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="valid")
        plt.legend()
        plt.show()

    def backtest(self, visualize=True):
        model = GRU(
            self.config["N_LAYER"],
            self.config["HIDDEN_DIM"],
            self.config["N_FEAT"],
            self.config["DROPOUT"],
        ).to(self.device)
        self.model = load_model(model, "result/best_model_weight_hb.pt", use_cuda=True)

        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        for i in range(0, self.test_x.shape[0], self.LEN_PRED):
            x = self.test_x[i][np.newaxis, :, :]
            out = self.model(x.float().cuda())[0]
            myWeights.append(out.detach().cpu().numpy())
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            myPortfolio.append(
                myPortfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy(), m_rtn))
            )
            equalPortfolio.append(
                equalPortfolio[-1] * np.exp(np.dot(EWPWeights, m_rtn))
            )

        idx = np.arange(0, len(self.test_date), self.LEN_PRED)
        performance = pd.DataFrame(
            {"EWP": equalPortfolio, "MyPortfolio": myPortfolio},
            index=np.array(self.test_date)[idx],
        )
        index_sp = pd.DataFrame(
            pd.read_csv("data/snp500_index.csv", index_col="Date")["Adj Close"]
        )
        index_sp = index_sp[self.test_date[0] :]
        performance["index_sp"] = index_sp["Adj Close"] * (
            myPortfolio[0] / index_sp["Adj Close"][0]
        )
        performance.to_csv("result/backtest.csv")

        if visualize:
            self._visualize_backtest(performance)
            self._visualize_weights(performance, myWeights)

        result = performance.copy()
        result["EWP_Return"] = np.log(result["EWP"]) - np.log(result["EWP"].shift(1))
        result["My_Return"] = np.log(result["MyPortfolio"]) - np.log(
            result["MyPortfolio"].shift(1)
        )
        result["Index_Return"] = np.log(result["index_sp"]) - np.log(
            result["index_sp"].shift(1)
        )
        result = result.dropna()

        expectedReturn = result[["EWP_Return", "My_Return", "Index_Return"]].mean()
        expectedReturn *= 12
        print("Annualized Return of Portfolio")
        print(expectedReturn)
        print("-" * 20)
        volatility = result[["EWP_Return", "My_Return", "Index_Return"]].std()
        volatility *= np.sqrt(12)
        print("Annualized Volatility of Portfolio")
        print(volatility)
        print("-" * 20)
        print("Annualized Sharp Ratio of Portfolio")
        print((expectedReturn / volatility))
        print("-" * 20)
        print("MDD")
        mdd_df = result[["EWP", "MyPortfolio", "index_sp"]].apply(self._get_mdd)
        print(mdd_df)

    def _visualize_backtest(self, performance):
        performance.plot(figsize=(14, 7), fontsize=10)
        plt.legend(fontsize=10)
        plt.savefig("result/performance.png")
        plt.show()

    def _visualize_weights(self, performance, weights):
        weights = np.array(weights)
        ticker = pd.read_csv("data/return_df.csv", index_col=0).columns
        n = self.N_STOCK
        plt.figure(figsize=(15, 10))
        for i in range(n):
            plt.plot(weights[:, i], label=ticker[i])
        plt.title("Weights")
        plt.xticks(
            np.arange(0, len(list(performance.index[1:]))),
            list(performance.index[1:]),
            rotation="vertical",
        )
        plt.legend()
        plt.savefig("result/weights.png")
        plt.show()

    def _get_mdd(self, x):
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return (
            x.index[peak_upper],
            x.index[peak_lower],
            (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper],
        )
