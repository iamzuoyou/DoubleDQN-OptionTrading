import math
import numpy as np
import torch
from Model.Env import Env
from Model.Double_DQN import Double_DQN
from Setting import arg
import time
import os

def train():
    state_dim = 10
    hidden_size = 64
    ModelType = 'Transformer'
    print(ModelType)
    agent = Double_DQN(state_dim=state_dim, hidden_size=hidden_size, ModelType=ModelType, MEMORY_THRESHOLD=500)

    epoch = 20
    seed_value = 1259097385086300
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    for k in range(epoch):
        env = Env(data_path15m="Data/CSI300-15m.csv",
                  data_path30m="Data/CSI300-30m.csv",
                  data_path60m="Data/CSI300-60m.csv")
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        done = 1
        while env.TimeCursor < 15570:  # The last day of the training set
            if done == 1:
                if total_timesteps != 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        total_timesteps, episode_num, episode_timesteps, episode_reward))
                    if env.DataLen - env.TimeCursor < arg.ADayTime * 5:
                        break

                done = 0
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            bar = env.Data.loc[env.TimeCursor, :]

            if done == 0:
                obs = env.Observation
                obs = obs.values
                obs = obs.astype(np.float64)

                obs15m = env.Observation15m
                obs15m = obs15m.values
                obs15m = obs15m.astype(np.float64)

                obs30m = env.Observation30m
                obs30m = obs30m.values
                obs30m = obs30m.astype(np.float64)

                obs60m = env.Observation60m
                obs60m = obs60m.values
                obs60m = obs60m.astype(np.float64)

                action = agent.action(obs,
                                     obs15m=obs15m,
                                     obs30m=obs30m,
                                     obs60m=obs60m,
                                     israndom=True, ResistancePointFlag=env.ResistancePointFlag, hold_time=env.hold_time)
                if env.hold_time > 112 or env.Observation.loc[env.Observation.shape[0] - 1, 'NextDay'] > 3:
                    action = 0

                new_obs, new15m, new30m, new60m, reward, done = env.step(action)

                new_obs = new_obs.values
                new_obs = new_obs.astype(np.float64)
                new15m = new15m.values
                new15m = new15m.astype(np.float64)
                new30m = new30m.values
                new30m = new30m.astype(np.float64)
                new60m = new60m.values
                new60m = new60m.astype(np.float64)

                episode_reward += reward
                agent.learn(obs, obs15m, obs30m, obs60m,
                            action, reward,
                            new_obs, new15m, new30m, new60m,
                            done, ResistancePointFlag=env.ResistancePointFlag, hold_time=env.hold_time)

                episode_timesteps += 1
                total_timesteps += 1
                timesteps_since_eval += 1

    # 保存模型到指定路径
    model_save_path = "ModelParam"
    os.makedirs(model_save_path, exist_ok=True)  # 确保目录存在
    now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    # 保存网络模型
    network_filename = f'{hidden_size}network-{ModelType}Noguiadance{now}.pth'
    torch.save(agent.network.state_dict(), os.path.join(model_save_path, network_filename))

    # 保存目标网络模型
    target_network_filename = f'{hidden_size}target-network-{ModelType}Noguiadance{now}.pth'
    torch.save(agent.target_network.state_dict(), os.path.join(model_save_path, target_network_filename))

    # 保存损失记录
    loss_filename = f'LossRecord{ModelType}Noguiadance{now}.npy'
    np.save(file=os.path.join(model_save_path, loss_filename), arr=np.array(agent.loss_record))

    print(f"Models saved to {model_save_path} with timestamp {now}.")
    print(f"Network file: {network_filename}")
    print(f"Target network file: {target_network_filename}")

if __name__ == "__main__":
    train()