import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)


    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (_, _) = self.rnn(x, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = torch.sigmoid(self.fc2(out))
        return out

    def init_hidden(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0
