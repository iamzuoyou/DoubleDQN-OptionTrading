import math
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from Model.Critic import Critic_AttentionCombine
from Model.Env import Env
from Model.Double_DQN_Load import Double_DQN
from Setting import arg
import os

def calculate_metrics(performance):
    """
    计算策略的各项指标
    """
    # 盈亏（P&L）
    pnl = performance['ProfitRate'].iloc[-1] - 1  # 最终收益减去初始资金

    # 年化收益（Annualized Return）
    total_days = (performance['Time'].iloc[-1] - performance['Time'].iloc[0]).days  # 总天数
    if total_days == 0:
        annualized_return = 0
    else:
        annualized_return = (performance['ProfitRate'].iloc[-1] ** (365 / total_days)) - 1

    # 盈亏比（Profit/Loss Ratio）
    profits = performance[performance['ProfitRate'].diff() > 0]['ProfitRate'].diff().sum()
    losses = performance[performance['ProfitRate'].diff() < 0]['ProfitRate'].diff().sum()
    profit_loss_ratio = abs(profits / losses) if losses != 0 else float('inf')

    # 胜率（Win Rate）
    win_rate = len(performance[performance['ProfitRate'].diff() > 0]) / len(performance) * 100

    # 最大回撤（Maximum Drawdown）
    performance['CumulativeMax'] = performance['ProfitRate'].cummax()
    performance['Drawdown'] = performance['ProfitRate'] - performance['CumulativeMax']
    max_drawdown = performance['Drawdown'].min()

    # 波动率（Volatility）
    returns = performance['ProfitRate'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # 年化波动率

    # 夏普比率（Sharpe Ratio）
    risk_free_rate = 0.025  # 无风险收益率设为2.5%
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else float('inf')

    # 输出指标
    metrics = {
        "盈亏 (P&L)": pnl,
        "年化收益 (Annualized Return)": f"{annualized_return * 100:.2f}%",
        "盈亏比 (Profit/Loss Ratio)": profit_loss_ratio,
        "胜率 (Win Rate)": f"{win_rate:.2f}%",
        "最大回撤 (Maximum Drawdown)": max_drawdown,
        "夏普比率 (Sharpe Ratio)": sharpe_ratio,
        "波动率 (Volatility)": volatility,
    }

    return metrics

def plot_performance(performance):
    """
    绘制收益曲线
    """
    plt.figure(figsize=(12, 6))
    plt.plot(performance['Time'], performance['ProfitRate'], label='Cumulative Profit')
    plt.title('Strategy Performance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate():
    seed_value = 1259097385086300
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

    state_dim = 10
    hidden_size = 64
    ModelType = 'Transformer'
    network = Critic_AttentionCombine(state_dim=state_dim, obs15m_dim=7, obs30m_dim=7, obs60m_dim=7, hiden_size=10)
    target_network = Critic_AttentionCombine(state_dim=state_dim, obs15m_dim=7, obs30m_dim=7, obs60m_dim=7, hiden_size=10)

    # 加载最新保存的模型
    model_save_path = "ModelParam"
    model_files = [f for f in os.listdir(model_save_path) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")

    # 找到最新的网络模型文件
    latest_network_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)))

    # 确保文件名是网络模型文件，而不是目标网络文件
    if "target-network" in latest_network_file:
        # 如果找到的是目标网络文件，尝试找到对应的网络模型文件
        latest_network_file = latest_network_file.replace("target-network", "network")

    # 加载网络模型
    network.load_state_dict(torch.load(os.path.join(model_save_path, latest_network_file), map_location=torch.device('cpu')))

    # 找到对应的目标网络文件
    target_network_file = latest_network_file.replace("network", "target-network")
    if not os.path.exists(os.path.join(model_save_path, target_network_file)):
        raise FileNotFoundError(f"Target network file not found: {target_network_file}")

    # 加载目标网络模型
    target_network.load_state_dict(torch.load(os.path.join(model_save_path, target_network_file), map_location=torch.device('cpu')))

    print(f"Loaded network model: {latest_network_file}")
    print(f"Loaded target network model: {target_network_file}")

    performance = pd.DataFrame(columns=['ProfitRate', 'Time'])
    agent = Double_DQN(network=network, target_network=target_network)

    epoch = 1
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
        while env.TimeCursor < env.DataLen - 2 * arg.ADayTime:

            new_data = dict()
            bar = env.Data.loc[env.TimeCursor, :]
            Assert = env.account.AllCash + env.account.getMarketValue(price=bar['close'], time=bar['time'], IV=env.HV)
            new_data['ProfitRate'] = Assert / env.account.initCash
            new_data['Time'] = bar['time']
            new_Data = pd.DataFrame(new_data, index=[0])
            performance = pd.concat([performance, new_Data], ignore_index=True, axis=0)

            if done == 1:
                if total_timesteps != 0:
                    print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                        total_timesteps, episode_num, episode_timesteps, episode_reward))
                    if env.DataLen - env.TimeCursor < arg.ADayTime * 5:
                        break

                done = 0  # Not begain
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if math.isnan(episode_reward):
                    print(episode_reward)
                    print(k)
                    print(env.Data.loc[env.TimeCursor, 'time'])

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
                                     israndom=False, ResistancePointFlag=env.ResistancePointFlag, hold_time=env.hold_time)
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

                episode_timesteps += 1
                total_timesteps += 1
                timesteps_since_eval += 1

    # 保存结果
    performance.to_csv('Performance.csv', index=False)

    # 计算指标
    metrics = calculate_metrics(performance)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 绘制收益曲线
    plot_performance(performance)

if __name__ == "__main__":
    evaluate()
