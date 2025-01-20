# DoubleDQN-OptionTrading: Straddle Option Trading Strategy based on Double DQN

## Introduction
Traditional option trading strategies often rely on static rules, which may not perform well in volatile markets. This project uses **Deep Reinforcement Learning (DRL)** to create a strategy that can learn from historical data and make decisions based on real-time market conditions. The strategy is designed to trade **straddle options**, which involve simultaneously buying a call and a put option on the same underlying asset.The code is inspired by the [AutomatedTradingSystem](https://github.com/JionAnderson/AutomatedTradingSystem) repository.

## Key Features
- **Multi-period feature extraction**: Combines 15-minute, 30-minute, and 60-minute market data to capture both short-term and long-term trends.
- **LSTM-Transformer hybrid architecture**: Uses LSTM for local temporal pattern extraction and Transformer for global dependency modeling.
- **Double DQN algorithm**: Improves upon traditional DQN by separating action selection and value estimation, reducing Q-value overestimation.
- **Dynamic risk management**: Utilizes the Black-Scholes model to dynamically adjust Delta values and position weights, ensuring effective risk control.
- **Priority Experience Replay (PER)**: Enhances learning efficiency by prioritizing experiences with higher TD errors.
- **Adaptive exploration strategy**: Balances exploration and exploitation using an adaptive ε-greedy policy.

### Usage
   ```bash
   python main.py
   ```
