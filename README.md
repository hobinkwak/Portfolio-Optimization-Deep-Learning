# Mean-Variance-Optimization-using-Deep-Learning

### Overview

- Mean-Variance Optimization to maximize Sharpe ratio using Deep Learning (PyTorch)
  - 1 layer GRU (or Transformer)
  - 1 FC layer
  - loss_fn : minimize negative sharpe (or Risk Parity)
  - optimizer : SAM (base SGD)


### DL Model

- GRU
- Transformer
  - the original source is [here](https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py)

### MVO

- Mean-Variance Optimization
- Maximize Annualized Sharpe Ratio

### Optimizer

- SAM optimizer (base optimizer : SGD with momentum 0.9) was used
  - the original source is [here](https://github.com/davda54/sam/blob/main/sam.py)
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

### Data

- As of December 27, 2021, stocks with more than 5,000 daily price data were selected.
  - AAPL, ABT, AMZN, CSCO, JPM, etc.
- Survivorship Bias
  - We didn't know in the past that these selected stocks would be in S&P500 until November 2021.
  - So, the performance might be different in real market

### Configuration

- train_config.json
```json
{
  "MODEL" : "GRU",
  "BATCH": 32,
  "SEED": 42,
  "EPOCHS" : 500,
  "EARLY_STOP" : 30,
  "LR" : 0.005,
  "USE_CUDA" : true,
  "N_LAYER": 1,
  "HIDDEN_DIM": 224,
  "N_HEAD" : 10,
  "N_FEAT": 50,
  "DROPOUT": 0.3
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
- Test Date : From **2017-04-11** To **2021-11-11**
- GRU (hidden_dim = 128), Dropout (0.3)
  - Expected Return : **0.366340** *(snp500 : 0.134716)*
  - Volatility : **0.224824** *(snp500 : 0.166945)*
  - Sharpe Ratio : **1.629453** *(snp500 : 0.806953)*
  - MDD : **-0.201779** *(snp500 : -0.233713)*

### Requirements
```
numpy==1.20.0
pandas==1.3.4
torch==1.7.1+cu110
yfinance==0.1.67
```
