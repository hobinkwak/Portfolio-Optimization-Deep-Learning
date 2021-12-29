# Portfolio-Optimization-with-Deep-Learning

## Overview

- 

## DL Model

- GRU
- Transformer
  - the original source is [here](https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py)

## MVO

- Mean-Variance Optimization
- Maximize Annualized Sharpe Ratio

## Optimizer

- SAM optimizer (base optimizer : Adam) was used
  - the original source is [here](https://github.com/davda54/sam/blob/main/sam.py)
- Learning Rate : 1e-4 (No Scheduler)

## Data

- As of November 30, 2021, stocks with more than 5,000 daily price data were selected.
  - AAPL, ABT, AMZN, CSCO, JPM, etc.
- Survivorship Bias
  - We didn't know in the past that these selected stocks would be in S&P500 until November 2021.
  - So, the performance might be different in real market

## Result

- 

## Requirements

- 
