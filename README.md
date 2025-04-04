# QuantumVIX: Adaptive Volatility RL Trading System

An advanced reinforcement learning-based trading system that specializes in exploiting volatility patterns in S&P 500 futures and options markets.

## Overview

QuantumVIX leverages a sophisticated combination of:

- **Black-Scholes volatility metrics** for options pricing and risk assessment
- **Multi-timeframe momentum signals** from both futures contracts and top S&P 500 components
- **Adaptive position sizing** based on real-time market risk conditions
- **Deep reinforcement learning** (PPO algorithm) for dynamic strategy optimization

The system continuously learns from market behavior, adjusting its trading parameters to adapt to changing volatility regimes. By focusing exclusively on S&P 500 instruments, QuantumVIX aims to capture alpha during both trending and range-bound market conditions while maintaining precise risk management through volatility-aware position sizing.

## Environment Setup

### Creating the Conda Environment

To set up the required environment, run the following commands:

```bash
# Create a new conda environment named qnt with Python 3.10
conda create -n qnt python=3.10

# Activate the environment
conda activate qnt

# Install PyTorch (adjust for your specific hardware if needed)
conda install pytorch -c pytorch

# Install the Quantiacs toolbox
pip install git+https://github.com/quantiacs/toolbox.git

# Install Stable Baselines3 with extras for RL
pip install "stable-baselines3[extra]"

# Install other dependencies
pip install gymnasium matplotlib pandas numpy scipy seaborn
```

### Environment Variables

The following environment variables can be set (optional):
- `DATA_BASE_URL`: Quantiacs data API URL (default: 'https://data-api.quantiacs.io/')
- `CACHE_RETENTION`: Number of days to keep cached data (default: '7')
- `CACHE_DIR`: Directory for data cache (default: 'data-cache')

## Project Structure

- `rl_options_strategy.py` - Main implementation of the RL trading strategy
- `models/` - Directory for saved model checkpoints
- `plots/` - Directory for performance visualizations
- `data-cache/` - Directory for cached market data

## Usage

To train and evaluate the model:

```bash
# Activate the environment
conda activate qnt

# Run the strategy
python rl_options_strategy.py
```

## Features

1. **Data Integration**
   - Real-time S&P 500 futures data
   - Top 10 S&P 500 components for broader market context
   - Historical volatility calculations

2. **Model Architecture**
   - PPO (Proximal Policy Optimization) reinforcement learning algorithm
   - Custom reward function based on risk-adjusted returns
   - State representation incorporating price, volatility, and momentum features

3. **Performance Monitoring**
   - Comprehensive visualization suite
   - Detailed logging of training metrics
   - Equity curve and drawdown tracking

## Development Status

The project is currently in active development. Recent updates include:
- Focusing exclusively on S&P 500 futures (F_ES)
- Adding comprehensive training monitoring with detailed metrics
- Implementing enhanced visualizations for model evaluation
- Fixing NaN rewards and other stability issues

## Files Description

- `rl_options_strategy.py`: Main strategy implementation with RL environment and training loop
- `training_log.txt`: Detailed training metrics and performance indicators
- Various visualization outputs in the `plots/` directory:
  - `training_progress.png`: Shows rewards, equity, and turnover during training
  - `training_equity_curve.png`: Shows the equity curve throughout training
  - `rl_model_performance.png`: Shows the final performance of the model 