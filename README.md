# Portfolio-Optimization-with-Deep-Learning

### Overview

- Mean-Variance Optimization to maximize Sharpe ratio using Deep Learning (PyTorch)
  - 1 layer GRU 
  - 1 FC layer
  - loss_fn : minimize negative sharpe
  - optimizer : SAM (base Adam)


### DL Model

- GRU
- Transformer
  - the original source is [here](https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py)

### MVO

- Mean-Variance Optimization
- Maximize Annualized Sharpe Ratio

### Optimizer

- SAM optimizer (base optimizer : Adam) was used
  - the original source is [here](https://github.com/davda54/sam/blob/main/sam.py)
- Learning Rate : 1e-4 (No Scheduler)

### Loss Function
```python
def max_sharpe(y_return, weights):
    weights = torch.unsqueeze(weights, 1) 
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)  
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]) 
    covmat = covmat.to('cuda')
    portReturn = torch.matmul(weights, meanReturn)  
    portVol = torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    objective = ((portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12)))
    return -objective.mean()
```

### Data

- As of November 30, 2021, stocks with more than 5,000 daily price data were selected.
  - AAPL, ABT, AMZN, CSCO, JPM, etc.
- Survivorship Bias
  - We didn't know in the past that these selected stocks would be in S&P500 until November 2021.
  - So, the performance might be different in real market

### Result
- Test Date : 
- GRU (hidden_dim = 64), Dropout (0.2)
  - Expected Return
  - Volatility
  - Sharpe Ratio
  - MDD
- GRU (hidden_dim = 128), Dropout (0.2)
  - Expected Return
  - Volatility
  - Sharpe Ratio
  - MDD

### Requirements
```
numpy==1.20.0
pandas==1.3.4
torch==1.7.1+cu110
yfinance==0.1.67
```
