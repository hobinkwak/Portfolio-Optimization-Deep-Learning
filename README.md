## Mean Variance Optimization by Gradient Descent (PyTorch)

### Overview

- Mean-Variance Optimization to maximize Sharpe ratio using Deep Learning (PyTorch)
  - 1 layer GRU / Transformer / TCN
  - 1 FC layer
  - loss_fn : minimize negative sharpe (or Risk Parity)
  - optimizer : SAM (base SGD)


### DL Model

- GRU
- Transformer
- TCN

### MVO

- Mean-Variance Optimization
- Maximize Annualized Sharpe Ratio

### Optimizer

- SAM optimizer (base optimizer : SGD with momentum 0.9) was used
  - the original source is [here](https://github.com/davda54/sam/blob/main/sam.py)
  - Of course, you can use Adam as base optimizer of SAM, or just Adam not SAM
    - but SAM optim (SGD) shows better performance than other options, empirically
- Learning Rate : 5e-3 (No Scheduler)

### Loss Function
```python
def max_sharpe(y_return, weights):
    weights = torch.unsqueeze(weights, 1) 
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)  
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to('cuda')
    portReturn = torch.matmul(weights, meanReturn)  
    portVol = torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    objective = ((portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12)))
    return -objective.mean()
```
### Constraint
- You can configure upper/lower bound for portfolio weights
  - this bounds are handled in **UB** and **LB** key in **train_config.json**
  - if you don't need any bounds, just set **LB=0** and **UB=1**
- portfolio weights are adjusted by the function below, before backpropagation
```python
def rebalance(self, weight, lb, ub):
    old = weight
    weight_clamped = torch.clamp(old, lb, ub)
    while True:
        leftover = (old - weight_clamped).sum().item()
        nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
        gift = leftover * (nominees / nominees.sum())
        weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
        old = weight_clamped
        if len(torch.where(weight_clamped > ub)[0]) == 0:
            break
        else:
            weight_clamped = torch.clamp(old, lb, ub)
    return weight_clamped
```

### Data

- As of December 27, 2021, stocks with more than 5,000 daily price data were selected.
  - AAPL, ABT, AMZN, CSCO, JPM, etc.
- Survivorship Bias (Look-ahead Bias)
  - We didn't know in the past that these selected stocks would be in S&P500 until November 2021.
  - So, the performance might (must) be different in real stock market
- Make Dataset for Training model
```bash
python dataload/data_download.py
python dataload/make_dataset.py
```

### Configuration

- train_config.json
```json
{
  "MODEL" : "GRU",
  "BATCH": 32,
  "SEED": 42,
  "EPOCHS" : 500,
  "EARLY_STOP" : 50,
  "LR" : 0.005,
  "MOMENTUM": 0.9,
  "USE_CUDA" : true,
  "N_LAYER": 1,
  "HIDDEN_DIM": 128,
  "N_HEAD" : 10,
  "N_FEAT": 50,
  "DROPOUT": 0.3,
  "BIDIRECTIONAL": false,
  "LB": 0,
  "UB": 0.2
}
```
- data_config.json
```json
{
  "START" : "2001-01-01",
  "END" : "2021-12-27",
  "N_STOCK" : 50,
  "LEN_DATA" : 5000,
  "TRAIN_LEN" : 63,
  "PRED_LEN" : 21,
  "TRAIN_RATIO": 0.75
}
```

### Result
- Test Date
  - From **2017-04-11** To **2021-11-11**
- Model
  - GRU (hidden_dim = 128), Dropout (0.3), Lower/Upper Bounds (0, 0.2)
- Performance (Transaction costs are **NOT** considered)
  - Expected Return : **0.310951** *(snp500 : 0.134716)*
  - Volatility : **0.180532** *(snp500 : 0.166945)*
  - Sharpe Ratio : **1.722411** *(snp500 : 0.806953)*
  - MDD : **-0.179543** *(snp500 : -0.233713)*
- You can see the cumulative return plot in **result** folder

### Requirements
```
numpy==1.20.0
pandas==1.3.4
torch==1.7.1+cu110
yfinance==0.1.67
```
