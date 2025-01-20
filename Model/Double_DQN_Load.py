import math
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from Model.Critic import Critic_LSTM, Critic_Transformer
from Setting import arg
from Model.Env import Env

# 检查设备可用性
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Test version of the DDQN framework
class Double_DQN():
    def __init__(self, network=None, target_network=None):
        self.network, self.target_network = network, target_network
        self.device = device
        self.network = self.network.to(device=self.device)
        self.network.eval()
        self.target_network = self.target_network.to(device=self.device)
        self.target_network.eval()
        self.ACTIONS_SIZE = 2

    def action(self, state,
               obs15m, obs30m, obs60m,
               israndom, ResistancePointFlag, hold_time):

        if israndom and random.random() < 0.1:
            return np.random.randint(0, self.ACTIONS_SIZE)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device=self.device)
        ResistancePointFlag = torch.tensor([[ResistancePointFlag]], dtype=torch.float32).to(device=self.device)

        obs15m = torch.unsqueeze(torch.FloatTensor(obs15m), 0).to(device=self.device)
        obs30m = torch.unsqueeze(torch.FloatTensor(obs30m), 0).to(device=self.device)
        obs60m = torch.unsqueeze(torch.FloatTensor(obs60m), 0).to(device=self.device)
        hold_time = torch.tensor([[hold_time]], dtype=torch.float32).to(device=self.device)
        actions_value = self.network.forward(state, ResistancePointFlag, hold_time, obs15m=obs15m, obs30m=obs30m, obs60m=obs60m).to(device=self.device)
        return torch.max(actions_value, 1)[1].cpu().data.numpy()[0]