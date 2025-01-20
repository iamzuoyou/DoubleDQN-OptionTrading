from torch import nn
import torch.nn.functional as F
from Setting import arg
from torch.autograd import Variable
import torch

# 检查设备可用性
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# This is the time series data module processed with lstm
class Critic_LSTM(nn.Module):
    def __init__(self, state_dim, hidden_size, num_layer=2):
        super(Critic_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.l2 = nn.Linear(hidden_size + 2, hidden_size)
        self.l3 = nn.Linear(hidden_size, 2)

    def forward(self, x, ResistancePointFlag, HoldTime):  # Add resistance levels and holding time information
        out, (h_n, c_n) = self.lstm(x)
        x1 = h_n
        x2 = x1[-1, :, :]
        x2 = torch.cat([x2, ResistancePointFlag, HoldTime], dim=1)
        x3 = self.l2(x2)
        x3 = F.leaky_relu(x3)
        x4 = self.l3(x3)
        return x4

# This is the time series data module processed with Transformer, The informative features used to learn the S_t^1
class Critic_Transformer(nn.Module):
    def __init__(self, state_dim, hidden_size, droupout=0.1):
        super(Critic_Transformer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=5, dim_feedforward=64, batch_first=True, dropout=droupout)
        self.l2 = nn.Linear(state_dim * arg.history_data_len * arg.ADayTime, hidden_size)
        self.l3 = nn.Linear(hidden_size + 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, ResistancePointFlag, HoldTime):  # Add resistance levels and holding time information
        h_n = self.encoder(x)
        x1 = torch.flatten(input=h_n, start_dim=1, end_dim=2)
        x2 = F.leaky_relu(self.l2(x1))
        x2 = torch.cat([x2, ResistancePointFlag, HoldTime], dim=1)
        x3 = F.leaky_relu(self.l3(x2))
        x4 = self.l4(x3)
        return x4

# This is the time series data module processed with Transformer, The informative features used to learn the obs_t^p
class Critic_TransformerKV(nn.Module):
    def __init__(self, state_dim, hidden_size, droupout=0.1):
        super(Critic_TransformerKV, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=1, dim_feedforward=64, batch_first=True, dropout=droupout)
        self.l2 = nn.Linear(state_dim * arg.history_data_len * arg.ADayTime, hidden_size)

    def forward(self, x):
        h_n = self.encoder(x)
        x1 = torch.flatten(input=h_n, start_dim=1, end_dim=2)
        x2 = F.leaky_relu(self.l2(x1))
        return x2

# This is the multi-period information fusion module, which integrates candlestick data from different time periods.
class Critic_AttentionCombine(nn.Module):
    def __init__(self, state_dim, obs15m_dim, obs30m_dim, obs60m_dim, hiden_size):
        super(Critic_AttentionCombine, self).__init__()
        self.Q = Critic_Transformer(state_dim, hiden_size)
        self.K15m = Critic_TransformerKV(obs15m_dim, hiden_size)
        self.K30m = Critic_TransformerKV(obs30m_dim, hiden_size)
        self.K60m = Critic_TransformerKV(obs60m_dim, hiden_size)
        self.QWK = Variable(torch.randn(3, hiden_size, hiden_size), requires_grad=True).to(device=device)
        self.out = nn.Linear(2 * hiden_size, 2)

    def forward(self, state, ResistancePointFlag, HoldTime, obs15m, obs30m, obs60m):
        # We query vector, and ready to compute the Q-vector attention score for each period
        q = self.Q.forward(state, ResistancePointFlag, HoldTime)
        q1 = torch.stack((q, q, q), dim=0)
        k15m = self.K15m.forward(obs15m).t()
        k30m = self.K30m.forward(obs30m).t()
        k60m = self.K60m.forward(obs60m).t()
        k1 = torch.stack((k15m, k30m, k60m), dim=0)
        # Calculate the attention score on the data for each period
        score = torch.bmm(torch.bmm(q1, self.QWK), k1)
        score = torch.diagonal(score, dim1=-1, dim2=-2)
        score = score.transpose(0, 1)
        score = score.unsqueeze(dim=-1)
        score = nn.functional.softmax(score, dim=-1)
        # The information of each period is fused with the query vector
        k1 = k1.transpose(0, -1)
        k2 = torch.bmm(k1, score)
        k2 = k2.squeeze(dim=-1)
        h = torch.concat([q, k2], dim=-1)
        Q_value = self.out(h)
        return Q_value