import math
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
from Model.Critic import Critic_LSTM, Critic_Transformer, Critic_AttentionCombine
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

# The deep reinforcement learning framework DDQN is used to update the neural network
class Double_DQN():
    def __init__(self, state_dim=10,
                 hidden_size=10,
                 ModelType='transformer',
                 BATCH_SIZE=16,
                 MEMORY_THRESHOLD=500):

        self.BATCH_SIZE = BATCH_SIZE
        self.LR = 0.01
        self.GAMMA = 0.99
        self.MEMORY_SIZE = 50000
        self.MEMORY_THRESHOLD = MEMORY_THRESHOLD
        self.UPDATE_TIME = 100
        self.ACTIONS_SIZE = 2

        if ModelType == 'lstm':
            self.network, self.target_network = Critic_LSTM(state_dim, hidden_size), Critic_LSTM(state_dim, hidden_size)
        else:
            self.network, self.target_network = Critic_AttentionCombine(state_dim=state_dim,
                                                                        obs15m_dim=7,
                                                                        obs30m_dim=7,
                                                                        obs60m_dim=7,
                                                                        hiden_size=10), \
                                                Critic_AttentionCombine(state_dim=state_dim,
                                                                        obs15m_dim=7,
                                                                        obs30m_dim=7,
                                                                        obs60m_dim=7,
                                                                        hiden_size=10),
        self.network.train(mode=True)
        self.target_network.train(mode=True)
        self.memory = deque()
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()
        self.loss_record = []
        self.device = device
        self.network = self.network.to(device=self.device)
        self.target_network = self.target_network.to(device=self.device)

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

    def learn(self, state, obs15m, obs30m, obs60m,
              action, reward,
              next_state, new15m, new30m, new60m,
              done, ResistancePointFlag, hold_time):

        self.memory.append((state, obs15m, obs30m, obs60m,
                            action, reward,
                            next_state, new15m, new30m, new60m,
                            1 - done, ResistancePointFlag, hold_time))

        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) < self.MEMORY_THRESHOLD:
            return

        if self.learning_count % self.UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.learning_count += 1

        batch = random.sample(self.memory, self.BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch]).to(device=self.device)
        obs15m = torch.FloatTensor([x[1] for x in batch]).to(device=self.device)
        obs30m = torch.FloatTensor([x[2] for x in batch]).to(device=self.device)
        obs60m = torch.FloatTensor([x[3] for x in batch]).to(device=self.device)

        action = torch.LongTensor([[x[4]] for x in batch]).to(device=self.device)
        reward = torch.FloatTensor([[x[5]] for x in batch]).to(device=self.device)

        next_state = torch.FloatTensor([x[6] for x in batch]).to(device=self.device)
        new15m = torch.FloatTensor([x[7] for x in batch]).to(device=self.device)
        new30m = torch.FloatTensor([x[8] for x in batch]).to(device=self.device)
        new60m = torch.FloatTensor([x[9] for x in batch]).to(device=self.device)

        done = torch.FloatTensor([[x[10]] for x in batch]).to(device=self.device)
        ResistancePointFlag = torch.FloatTensor([[x[11]] for x in batch]).to(device=self.device)
        hlod_time = torch.FloatTensor([[x[12]] for x in batch]).to(device=self.device)

        actions_value = self.network.forward(next_state, ResistancePointFlag, hlod_time,
                                             obs15m=new15m, obs30m=new30m, obs60m=new60m)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1)
        eval_q = self.network.forward(state, ResistancePointFlag, hlod_time,
                                      obs15m=obs15m, obs30m=obs30m, obs60m=obs60m).gather(1, action)
        next_q = self.target_network.forward(next_state, ResistancePointFlag, hlod_time,
                                             obs15m=new15m, obs30m=new30m, obs60m=new60m).gather(1, next_action)
        target_q = reward + self.GAMMA * next_q * done

        loss = self.loss_func(eval_q, target_q)
        self.loss_record.append(loss.cpu().item())

        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()