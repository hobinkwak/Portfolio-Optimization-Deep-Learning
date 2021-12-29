import os
import json
import random
import numpy as np
import torch
from train.train import Trainer


def work(config, visualize=False):
    worker = Trainer(config)
    worker.set_data()
    worker.train(visualize)
    worker.backtest(visualize)


if __name__ == "__main__":
    config = json.load(open("config/train_config.json", "r"))
    os.environ["PYTHONHASHSEED"] = str(config["SEED"])
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    work(config, visualize=True)
