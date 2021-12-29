import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_stocks, dropout_p=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            n_stocks, self.hidden_dim, num_layers=self.n_layers, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(self.hidden_dim, n_stocks)
        self.swish = nn.SiLU()

    def forward(self, x):
        init_h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to('cuda')
        x, _ = self.gru(x, init_h)
        h_t = x[:, -1, :]
        logit = self.fc(self.dropout(h_t))
        logit = self.swish(logit)
        return F.softmax(logit, dim=-1)


if __name__ == "__main__":
    pass
