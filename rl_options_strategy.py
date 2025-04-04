import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Import Quantiacs libraries
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta
import qnt.stats as qnstats
import qnt.graph as qngraph

def create_training_logger():
    """Create a logger for detailed training metrics"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_log.txt', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('training_logger')

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class OptionsTraderEnv(gym.Env):
    """
    Custom reinforcement learning environment for options trading
    using momentum and Black-Scholes information
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, sp500_data, futures_data, top_stocks_data, 
                start_idx=0, window_size=60, 
                training_period=(2006, 2023), testing=False, render_mode="human"):
        super(OptionsTraderEnv, self).__init__()
        
        # Store data
        self.sp500_data = sp500_data
        self.futures_data = futures_data
        self.top_stocks_data = top_stocks_data
        self.render_mode = render_mode
        
        # Environment parameters
        self.window_size = window_size
        self.start_idx = start_idx
        self.current_step = self.start_idx
        self.training_period = training_period
        self.testing = testing
        
        # Determine available assets (futures contracts)
        self.assets = self.futures_data.coords['asset'].values
        self.num_assets = len(self.assets)
        
        # Action space: For each asset, decide position [-1, 0, 1]
        # -1 = short, 0 = no position, 1 = long
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation space: Features for each asset + S&P 500 + top stocks
        # For each asset: [price, vol, hist_vol, bs_call, bs_put, momentum1, momentum2, ...]
        # For S&P 500: [price, momentum, volatility, ...]
        # For top stocks: [price1, momentum1, price2, momentum2, ...]
        num_features_per_asset = 10  # Price, volatility metrics, BS metrics, momentum metrics
        num_sp500_features = 5  # S&P 500 specific features
        num_top_stock_features = 5 * 10  # 5 features for each of the top 10 stocks
        
        total_features = (self.num_assets * num_features_per_asset) + num_sp500_features + num_top_stock_features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)
        
        # Portfolio state
        self.weights = np.zeros(self.num_assets)
        self.equity = 1.0  # Starting with $1M normalized to 1.0
        self.returns = []
        
        # Performance tracking
        self.episode_returns = []
        self.sharpe_ratio = 0
        self.max_drawdown = 0

    def _get_safe_scalar(self, value, default=0.0):
        """Safely convert a value to scalar float"""
        try:
            # Handle array-like inputs
            if hasattr(value, '__len__') and len(value) == 1:
                return float(value.item() if hasattr(value, 'item') else value[0])
            # Handle scalar inputs
            return float(value)
        except (TypeError, ValueError):
            return default
        
    def _get_time_idx(self, step):
        """Map step to actual time index in the data"""
        # Map the current step to the appropriate date in the data
        # Only use dates within the training/testing periods
        available_dates = self.futures_data.time.values
        
        if self.testing:
            # For testing, use 2024-2025 data
            mask = (pd.to_datetime(available_dates).year >= 2024)
        else:
            # For training, use data from training_period[0] to training_period[1]
            mask = ((pd.to_datetime(available_dates).year >= self.training_period[0]) & 
                    (pd.to_datetime(available_dates).year <= self.training_period[1]))
        
        valid_dates = available_dates[mask]
        
        # Ensure step is within range
        if step >= len(valid_dates):
            step = len(valid_dates) - 1
        
        return valid_dates[step]
    
    def _get_state(self):
        """
        Get current state observation consisting of:
        - Features for each futures contract (price, vol, historical vol, BS metrics, momentum)
        - S&P 500 features (price, momentum, volatility)
        - Top stocks features (price, momentum)
        """
        # Get current time index
        current_time = self._get_time_idx(self.current_step)
        
        # Get a window of historical data ending at current_time
        end_time = current_time
        
        # Fix the issue with numpy.datetime64 handling
        futures_times = self.futures_data.time.values
        closest_idx = 0
        min_diff = float('inf')
        for i, t in enumerate(futures_times):
            diff = abs(pd.Timestamp(t) - pd.Timestamp(current_time)).total_seconds()
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        start_idx = max(0, closest_idx - self.window_size)
        start_time = futures_times[start_idx]
        
        # Extract futures data for this time window
        futures_window = self.futures_data.sel(time=slice(start_time, end_time))
        
        # Extract S&P 500 data
        if self.sp500_data is not None:
            try:
                sp500_window = self.sp500_data.sel(time=slice(start_time, end_time))
            except (KeyError, ValueError):
                # If exact dates don't match, create empty placeholder
                sp500_window = None
        else:
            sp500_window = None
        
        # Extract top stocks data
        if self.top_stocks_data is not None:
            try:
                stocks_window = self.top_stocks_data.sel(time=slice(start_time, end_time))
            except (KeyError, ValueError):
                stocks_window = None
        else:
            stocks_window = None
        
        # Initialize observation array
        features = []
        
        # Process each futures contract
        for asset in self.assets:
            try:
                # Extract asset data
                asset_data = futures_window.sel(asset=asset)
                
                # Get basic price and volume data
                try:
                    close_values = asset_data.sel(field="close").values
                    if len(close_values) > 0:
                        close = self._get_safe_scalar(close_values[-1])
                    else:
                        close = 0.0
                except (IndexError, KeyError):
                    close = 0.0
                
                # Handle missing or nan values
                if np.isnan(close):
                    close = 0.0
                
                # Calculate historical volatility
                hist_vol = self._calculate_historical_volatility(asset_data.sel(field="close").values)
                
                # Calculate Black-Scholes metrics
                bs_call, bs_put = self._calculate_bs_metrics(close, hist_vol)
                
                # Calculate momentum metrics
                mom_1d = self._calculate_momentum(asset_data.sel(field="close").values, period=1)
                mom_5d = self._calculate_momentum(asset_data.sel(field="close").values, period=5)
                mom_20d = self._calculate_momentum(asset_data.sel(field="close").values, period=20)
                
                # Calculate volume momentum
                try:
                    vol_data = asset_data.sel(field="vol").values
                    vol_mom = self._calculate_volume_momentum(vol_data)
                except (IndexError, KeyError):
                    vol_mom = 0.0
                
                # Option metrics
                option_volatility_spread = self._calculate_option_vol_spread(hist_vol)
                
                # Convert to scalar values if necessary
                close = self._get_safe_scalar(close)
                hist_vol = self._get_safe_scalar(hist_vol)
                bs_call = self._get_safe_scalar(bs_call)
                bs_put = self._get_safe_scalar(bs_put)
                mom_1d = self._get_safe_scalar(mom_1d)
                mom_5d = self._get_safe_scalar(mom_5d)
                mom_20d = self._get_safe_scalar(mom_20d)
                vol_mom = self._get_safe_scalar(vol_mom)
                option_volatility_spread = self._get_safe_scalar(option_volatility_spread)
                
                # Get index of this asset in weights array
                asset_idx = np.where(self.assets == asset)[0][0]
                current_weight = self._get_safe_scalar(self.weights[asset_idx])
                
                # Compile features for this asset
                asset_features = [
                    close, hist_vol, bs_call, bs_put, 
                    mom_1d, mom_5d, mom_20d, vol_mom,
                    option_volatility_spread, current_weight
                ]
                
                features.extend(asset_features)
                
            except (KeyError, ValueError, IndexError) as e:
                # If asset data is missing, use zeros
                features.extend([0.0] * 10)
        
        # Add S&P 500 features
        if sp500_window is not None:
            try:
                try:
                    sp500_close_values = sp500_window.sel(field="close").values
                    sp500_close = self._get_safe_scalar(sp500_close_values[-1] if len(sp500_close_values) > 0 else 0.0)
                except (IndexError, KeyError):
                    sp500_close = 0.0
                
                sp500_hist_vol = self._calculate_historical_volatility(sp500_window.sel(field="close").values)
                sp500_mom_1d = self._calculate_momentum(sp500_window.sel(field="close").values, period=1)
                sp500_mom_5d = self._calculate_momentum(sp500_window.sel(field="close").values, period=5)
                sp500_mom_20d = self._calculate_momentum(sp500_window.sel(field="close").values, period=20)
                
                # Convert to scalar values if necessary
                sp500_features = [
                    self._get_safe_scalar(sp500_close),
                    self._get_safe_scalar(sp500_hist_vol),
                    self._get_safe_scalar(sp500_mom_1d),
                    self._get_safe_scalar(sp500_mom_5d),
                    self._get_safe_scalar(sp500_mom_20d)
                ]
                
                features.extend(sp500_features)
            except (KeyError, ValueError, IndexError) as e:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
        
        # Add top stocks features
        if stocks_window is not None:
            try:
                stock_count = 0
                for stock in stocks_window.coords['asset'].values:
                    if stock_count >= 10:  # Only use top 10 stocks
                        break
                        
                    stock_data = stocks_window.sel(asset=stock)
                    
                    try:
                        stock_close_values = stock_data.sel(field="close").values
                        stock_close = self._get_safe_scalar(stock_close_values[-1] if len(stock_close_values) > 0 else 0.0)
                    except (IndexError, KeyError):
                        stock_close = 0.0
                        
                    stock_hist_vol = self._calculate_historical_volatility(stock_data.sel(field="close").values)
                    stock_mom_1d = self._calculate_momentum(stock_data.sel(field="close").values, period=1)
                    stock_mom_5d = self._calculate_momentum(stock_data.sel(field="close").values, period=5)
                    stock_mom_20d = self._calculate_momentum(stock_data.sel(field="close").values, period=20)
                    
                    # Ensure all values are scalar
                    stock_features = [
                        self._get_safe_scalar(stock_close),
                        self._get_safe_scalar(stock_hist_vol),
                        self._get_safe_scalar(stock_mom_1d),
                        self._get_safe_scalar(stock_mom_5d),
                        self._get_safe_scalar(stock_mom_20d)
                    ]
                    
                    features.extend(stock_features)
                    stock_count += 1
                
                # If we didn't get 10 stocks, fill in the rest with zeros
                missing_stocks = 10 - stock_count
                if missing_stocks > 0:
                    features.extend([0.0] * (missing_stocks * 5))
                    
            except (KeyError, ValueError, IndexError) as e:
                # Fill in missing data
                missing_features = 10 * 5
                features.extend([0.0] * missing_features)
        else:
            missing_features = 10 * 5
            features.extend([0.0] * missing_features)
                    
        # Normalize and ensure no NaNs
        features = np.array(features, dtype=np.float32)
        features[np.isnan(features)] = 0.0
        features[np.isinf(features)] = 0.0
        
        # Simple normalization
        # Scale to zero mean and unit variance where possible
        nonzero_indices = np.where(np.abs(features) > 1e-10)[0]
        if len(nonzero_indices) > 0:
            features[nonzero_indices] = (features[nonzero_indices] - np.mean(features[nonzero_indices])) / (np.std(features[nonzero_indices]) + 1e-10)
        
        return features
    
    def _calculate_historical_volatility(self, prices, window=20):
        """Calculate historical volatility from price data"""
        if len(prices) <= window:
            return 0
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Handle nans and infs
        log_returns = np.nan_to_num(log_returns, nan=0, posinf=0, neginf=0)
        
        # Calculate volatility (annualized)
        if len(log_returns) <= 1:
            return 0
        
        volatility = np.std(log_returns[-window:]) * np.sqrt(252)
        return volatility
    
    def _calculate_momentum(self, prices, period=20):
        """Calculate momentum as percentage price change over period"""
        if len(prices) <= period:
            return 0
        
        if prices[-period] == 0 or np.isnan(prices[-period]):
            return 0
        
        return (prices[-1] / prices[-period]) - 1
    
    def _calculate_volume_momentum(self, volumes, period=10):
        """Calculate volume momentum as ratio of current to average volume"""
        if len(volumes) <= period:
            return 0
        
        recent_volumes = volumes[-period:]
        nonzero_vols = recent_volumes[recent_volumes > 0]
        
        if len(nonzero_vols) == 0:
            return 0
        
        avg_volume = np.mean(nonzero_vols)
        if avg_volume == 0:
            return 0
        
        return volumes[-1] / avg_volume
    
    def _calculate_bs_metrics(self, price, volatility, time_to_expiry=30/252, risk_free_rate=0.02):
        """Calculate Black-Scholes option prices"""
        if price <= 0 or volatility <= 0:
            return 0, 0
        
        # At-the-money options
        strike = price
        
        # Calculate d1 and d2
        try:
            d1 = (np.log(price/strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Calculate call and put prices
            call_price = price * stats.norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)
            put_price = strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2) - price * stats.norm.cdf(-d1)
            
            return call_price, put_price
        except (ValueError, ZeroDivisionError):
            return 0, 0
    
    def _calculate_option_vol_spread(self, hist_vol, baseline=0.2):
        """
        Calculate a proxy for the spread between implied vol and historical vol
        Since we don't have actual implied vol data, we approximate it
        """
        if hist_vol <= 0:
            return 0
        
        # Simple model: high historical vol tends to make implied vol higher with a spread
        implied_vol_approx = hist_vol * 1.1 + 0.02  # Typically implied > historical
        return implied_vol_approx - hist_vol
    
    def _calculate_reward(self, new_weights):
        """
        Calculate reward based on returns and risk-adjusted metrics
        Rewards positive returns but penalizes excessive risk
        """
        # Get current and next time indices
        current_time = self._get_time_idx(self.current_step)
        next_time = self._get_time_idx(self.current_step + 1)
        
        # Extract price data
        try:
            current_prices = self.futures_data.sel(time=current_time, field="close")
            next_prices = self.futures_data.sel(time=next_time, field="close")
            
            # Calculate returns for each asset
            asset_returns = (next_prices / current_prices - 1).values
            
            # Replace nans and infs with zeros
            asset_returns = np.nan_to_num(asset_returns, nan=0, posinf=0, neginf=0)
            
            # Calculate portfolio return
            portfolio_return = np.sum(new_weights * asset_returns)
            
            # Store return for tracking
            self.returns.append(portfolio_return)
            
            # Include risk penalty: penalize large drawdowns and excessive volatility
            if len(self.returns) > 20:
                recent_returns = np.array(self.returns[-20:])
                volatility_penalty = -0.1 * np.std(recent_returns) * np.sqrt(252)  # Annualized volatility
                
                # Drawdown penalty
                equity_curve = np.cumprod(1 + np.array(self.returns))
                peak = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve / peak) - 1
                max_drawdown = np.min(drawdown)
                drawdown_penalty = -0.2 * abs(max_drawdown)
                
                # Turnover penalty (to discourage excessive trading)
                turnover = np.sum(np.abs(new_weights - self.weights))
                turnover_penalty = -0.05 * turnover
                
                # Net reward
                reward = portfolio_return + volatility_penalty + drawdown_penalty + turnover_penalty
            else:
                # Not enough history for risk metrics
                reward = portfolio_return
            
            # Update equity
            self.equity *= (1 + portfolio_return)
            
            return reward
            
        except (KeyError, ValueError):
            # If data is missing, return small negative reward
            return -0.01
    
    def step(self, action):
        """
        Take an action in the environment
        Action represents the target weights for each asset
        """
        # Convert action to portfolio weights
        raw_weights = action.copy()
        
        # Normalize weights to sum to 1.0 (absolute sum)
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum > 1e-8:
            normalized_weights = raw_weights / abs_sum
        else:
            normalized_weights = np.zeros_like(raw_weights)
        
        # Save previous weights for turnover calculation
        self._previous_weights = self.weights.copy()
        
        # Calculate reward based on these weights
        reward = self._calculate_reward(normalized_weights)
        
        # Update weights
        self.weights = normalized_weights
        
        # Move to next step
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state()
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        # Episode terminates if:
        # 1. Equity falls below a threshold (e.g., 50% of initial capital)
        # 2. Reached the end of the training/testing period
        current_time = self._get_time_idx(self.current_step)
        
        if self.equity < 0.5:
            terminated = True  # Bankrupt
        
        if self.testing:
            # For testing, done when we reach the end of 2025
            truncated = truncated or pd.to_datetime(current_time).year > 2025
        else:
            # For training, done when we reach the end of training_period[1]
            truncated = truncated or pd.to_datetime(current_time).year > self.training_period[1]
        
        info = {
            'equity': self.equity,
            'weights': self.weights,
            'time': current_time,
            'portfolio_return': self.returns[-1] if len(self.returns) > 0 else 0.0,
            'r': reward  # Add raw reward to info dict for callback tracking
        }
        
        return next_state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset to the beginning of the time period
        self.current_step = self.start_idx
        
        # Reset portfolio state
        self.weights = np.zeros(self.num_assets)
        self._previous_weights = np.zeros(self.num_assets)  # Initialize previous weights for turnover calc
        self.equity = 1.0
        self.returns = []
        
        # Get initial state
        initial_state = self._get_state()
        
        # Info dictionary for reset
        info = {}
        
        return initial_state, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Show current portfolio state and performance
            print(f"Step: {self.current_step}, Equity: {self.equity:.2f}")
            print(f"Current Weights: {self.weights}")
            
            # If enough returns, show performance metrics
            if len(self.returns) > 20:
                annual_return = np.mean(self.returns) * 252
                volatility = np.std(self.returns) * np.sqrt(252)
                sharpe = annual_return / volatility if volatility > 0 else 0
                
                print(f"Annualized Return: {annual_return:.2%}")
                print(f"Volatility: {volatility:.2%}")
                print(f"Sharpe Ratio: {sharpe:.2f}")
        
        return None

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving the model when training reward improves
    """
    def __init__(self, check_freq=1000, log_dir="./models", verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.training_history = {
            'timesteps': [],
            'rewards': [],
            'equity': [],
            'portfolio_turnover': [],
            'mean_rewards': []
        }
        # Create the plot directory
        os.makedirs('plots', exist_ok=True)
        # Initialize logger
        self.training_logger = create_training_logger()
    
    def _init_callback(self):
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_step(self):
        # Store raw episode rewards from env for plotting
        if hasattr(self.model, 'env') and hasattr(self.model.env.envs[0], 'returns'):
            env = self.model.env.envs[0]
            if len(env.returns) > 0:
                self.training_history['rewards'].append(env.returns[-1])
                self.training_history['equity'].append(env.equity)
                
                # Calculate portfolio turnover if we have previous weights
                if hasattr(env, 'weights') and hasattr(env, '_previous_weights'):
                    turnover = np.sum(np.abs(env.weights - env._previous_weights))
                    self.training_history['portfolio_turnover'].append(turnover)
                else:
                    self.training_history['portfolio_turnover'].append(0)
                
                self.training_history['timesteps'].append(self.num_timesteps)
                
                # Log detailed metrics every 100 steps
                if self.n_calls % 100 == 0:
                    self.training_logger.info(f"Step: {self.num_timesteps}, " 
                                    f"Return: {env.returns[-1]:.4f}, "
                                    f"Equity: {env.equity:.4f}, "
                                    f"Turnover: {turnover:.4f}")
        
        if self.n_calls % self.check_freq == 0:
            # Calculate mean reward safely
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                self.training_history['mean_rewards'].append(mean_reward)
                
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Mean reward: {mean_reward:.2f}")
                
                # Log detailed performance statistics
                self.training_logger.info(f"CHECKPOINT - Timesteps: {self.num_timesteps}")
                self.training_logger.info(f"Mean reward: {mean_reward:.4f}")
                
                # If we have enough history, calculate more metrics
                if len(self.training_history['rewards']) > 20:
                    recent_rewards = self.training_history['rewards'][-20:]
                    sharpe = np.mean(recent_rewards) / (np.std(recent_rewards) + 1e-6) * np.sqrt(252)
                    self.training_logger.info(f"Recent Sharpe Ratio: {sharpe:.4f}")
                    
                    # Calculate drawdown
                    if len(self.training_history['equity']) > 0:
                        equity_array = np.array(self.training_history['equity'])
                        peak = np.maximum.accumulate(equity_array)
                        drawdown = (equity_array / peak) - 1
                        max_dd = np.min(drawdown)
                        self.training_logger.info(f"Current Drawdown: {drawdown[-1]:.4f}")
                        self.training_logger.info(f"Max Drawdown: {max_dd:.4f}")
                
                # Save model if improved
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model with reward {mean_reward:.2f}")
                    self.model.save(self.save_path)
                    self.training_logger.info(f"NEW BEST MODEL - Mean reward: {mean_reward:.4f}")
            else:
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print("No complete episodes yet, skipping reward calculation")
                self.training_logger.info(f"No complete episodes at timestep {self.num_timesteps}")
            
            # Create and save training progress plots
            self._plot_training_progress()
        
        return True
    
    def _plot_training_progress(self):
        """Create visualizations of training progress"""
        # Create a multi-panel plot to track various metrics
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot episode rewards
        if len(self.training_history['timesteps']) > 0 and len(self.training_history['rewards']) > 0:
            axes[0].plot(self.training_history['timesteps'], self.training_history['rewards'], 'b-', alpha=0.7)
            if len(self.training_history['rewards']) > 100:
                # Add a rolling mean to see trends more clearly
                window_size = min(100, len(self.training_history['rewards']) // 5)
                rolling_mean = np.convolve(self.training_history['rewards'], 
                                           np.ones(window_size)/window_size, mode='valid')
                rolling_steps = self.training_history['timesteps'][window_size-1:]
                if len(rolling_mean) == len(rolling_steps):
                    axes[0].plot(rolling_steps, rolling_mean, 'r-', linewidth=2)
            
            axes[0].set_title('Episode Rewards')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True)
        
        # Plot equity curve
        if len(self.training_history['timesteps']) > 0 and len(self.training_history['equity']) > 0:
            axes[1].plot(self.training_history['timesteps'], self.training_history['equity'], 'g-')
            axes[1].set_title('Portfolio Equity')
            axes[1].set_ylabel('Equity')
            axes[1].grid(True)
        
        # Plot portfolio turnover
        if len(self.training_history['timesteps']) > 0 and len(self.training_history['portfolio_turnover']) > 0:
            axes[2].plot(self.training_history['timesteps'], self.training_history['portfolio_turnover'], 'r-')
            axes[2].set_title('Portfolio Turnover')
            axes[2].set_ylabel('Turnover')
            axes[2].set_xlabel('Timesteps')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'training_progress.png'))
        plt.close(fig)
        
        # Log current stats
        if len(self.training_history['rewards']) > 0:
            recent_rewards = self.training_history['rewards'][-min(100, len(self.training_history['rewards'])):]
            print(f"Recent mean reward: {np.mean(recent_rewards):.4f}")
            print(f"Recent reward volatility: {np.std(recent_rewards):.4f}")
            print(f"Current equity: {self.training_history['equity'][-1]:.4f}")

def black_scholes_d1(S, K, T, r, sigma):
    """Calculate d1 parameter in Black-Scholes formula"""
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = black_scholes_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def calculate_historical_volatility(prices, window=30):
    """Calculate historical volatility using log returns"""
    if isinstance(prices, xr.DataArray):
        # Convert to numpy array
        prices_np = prices.values
    else:
        prices_np = prices
    
    # Calculate log returns
    log_returns = np.diff(np.log(prices_np), prepend=np.log(prices_np[0]))
    
    # Handle NaNs
    log_returns = np.nan_to_num(log_returns, nan=0)
    
    # Calculate rolling volatility
    volatility = np.zeros_like(prices_np)
    for i in range(len(prices_np)):
        if i < window:
            volatility[i] = np.std(log_returns[:i+1]) * np.sqrt(252) if i > 0 else 0
        else:
            volatility[i] = np.std(log_returns[i-window+1:i+1]) * np.sqrt(252)
    
    return volatility

def create_rl_weights(model, env, futures_data, start_date="2024-01-01", end_date="2025-04-01"):
    """
    Use a trained RL model to generate weights for the testing period
    """
    # Create a test environment
    test_env = OptionsTraderEnv(
        sp500_data=env.sp500_data,
        futures_data=futures_data,
        top_stocks_data=env.top_stocks_data,
        testing=True
    )
    
    # Reset the environment
    obs, _ = test_env.reset()
    
    # Get time range
    available_times = futures_data.time.values
    start_idx = 0
    for i, time in enumerate(available_times):
        if pd.to_datetime(time) >= pd.to_datetime(start_date):
            start_idx = i
            break
    
    # Generate weights for each date in the testing period
    all_weights = []
    all_times = []
    test_env.current_step = start_idx
    
    done = False
    while not done:
        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # Store weights and time
        all_weights.append(test_env.weights.copy())
        all_times.append(info['time'])
        
        # Check if episode is done
        done = terminated or truncated
        
        # Check if we've reached the end date
        if pd.to_datetime(info['time']) >= pd.to_datetime(end_date):
            break
    
    # Convert to xarray DataArray
    weights_array = np.array(all_weights)
    
    # Create dimensions
    weights_xr = xr.DataArray(
        data=weights_array,
        dims=["time", "asset"],
        coords={
            "time": all_times,
            "asset": futures_data.coords["asset"].values
        }
    )
    
    return weights_xr

def plot_performance(returns, weights, title="RL Model Performance"):
    """Plot the performance of the RL model"""
    # Create equity curve
    equity = (1 + returns).cumprod()
    
    # Calculate performance metrics
    annual_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Calculate drawdowns
    peak = equity.expanding(min_periods=1).max()
    drawdown = (equity / peak) - 1
    max_drawdown = drawdown.min()
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 2]})
    
    # Plot equity curve
    ax1.plot(equity.index, equity.values)
    ax1.set_title(f"{title} - Equity Curve")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True)
    
    # Add performance metrics as text
    metrics_text = (
        f"Annual Return: {annual_return:.2%}\n"
        f"Volatility: {volatility:.2%}\n"
        f"Sharpe Ratio: {sharpe:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2%}"
    )
    ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot drawdowns
    ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
    ax2.set_title("Drawdowns")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True)
    
    # Plot position heat map (top 10 assets by average absolute weight)
    # Identify top assets
    mean_abs_weights = np.mean(np.abs(weights.values), axis=0)
    top_indices = np.argsort(mean_abs_weights)[-10:]
    top_assets = [weights.coords["asset"].values[i] for i in top_indices]
    
    # Create heatmap data
    heatmap_data = weights.loc[:, top_assets].values.T
    
    # Plot heatmap
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                yticklabels=top_assets,
                xticklabels=[i.strftime('%Y-%m') if i % 100 == 0 else "" 
                             for i, t in enumerate(weights.coords["time"].values)],
                ax=ax3)
    ax3.set_title("Position Weights for Top 10 Assets")
    ax3.set_ylabel("Asset")
    ax3.set_xlabel("Time")
    
    plt.tight_layout()
    plt.savefig("plots/rl_model_performance.png")
    plt.close()

def main():
    print("Loading data...")
    
    # Load futures data for training the RL model
    all_futures_data = qndata.futures.load_data(min_date="2006-01-01")
    
    # Filter to only include SP500-related futures contracts
    sp500_futures_symbols = ['F_ES', 'F_SP']  # E-mini S&P 500 and S&P 500 futures
    
    # Check if these assets exist in the data
    available_assets = all_futures_data.coords['asset'].values
    sp500_futures_assets = [asset for asset in sp500_futures_symbols if asset in available_assets]
    
    if not sp500_futures_assets:
        print("Warning: Could not find S&P 500 futures. Using all futures instead.")
        futures_data = all_futures_data
    else:
        print(f"Using S&P 500 futures: {sp500_futures_assets}")
        # Filter the futures data to only include S&P 500 related contracts
        futures_data = all_futures_data.sel(asset=sp500_futures_assets)
    
    # Try to load S&P 500 data
    try:
        print("Loading S&P 500 data...")
        sp500_data = qndata.stocks.load_spx_data(min_date="2006-01-01", max_date="2025-04-01")
        
        # Get top 10 stocks by market cap
        sp500_list = qndata.stocks.load_spx_list(min_date="2006-01-01", max_date="2025-04-01")
        top_stock_ids = [stock['id'] for stock in sp500_list[:10]]
        
        print(f"Top 10 stocks: {top_stock_ids}")
        
        # Load data for top stocks
        top_stocks_data = qndata.stocks.load_spx_data(
            min_date="2006-01-01", 
            max_date="2025-04-01",
            assets=top_stock_ids
        )
    except Exception as e:
        print(f"Error loading stock data: {e}")
        sp500_data = None
        top_stocks_data = None
    
    print("Creating training environment...")
    
    # Create environment
    env = OptionsTraderEnv(
        sp500_data=sp500_data,
        futures_data=futures_data,
        top_stocks_data=top_stocks_data,
        training_period=(2006, 2023)
    )
    
    # Wrap environment for SB3
    vec_env = DummyVecEnv([lambda: env])
    
    # Create callback for saving best model and monitoring training
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="./models", verbose=1)
    
    # Check if model exists
    model_path = "./models/best_model.zip"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("Training new model...")
        # Create RL model with tracking of value function and entropy
        model = PPO("MlpPolicy", vec_env, 
                    learning_rate=0.0003, 
                    n_steps=2048,
                    batch_size=64,
                    ent_coef=0.01,
                    verbose=1)
        
        # Print training start message
        print("\n" + "="*50)
        print("STARTING MODEL TRAINING")
        print("="*50)
        print(f"Asset universe: {env.assets}")
        print(f"Number of assets: {env.num_assets}")
        print(f"Training period: {env.training_period[0]}-{env.training_period[1]}")
        print(f"Observation space shape: {env.observation_space.shape}")
        print(f"Action space shape: {env.action_space.shape}")
        print("="*50 + "\n")
        
        # Train model
        total_timesteps = 1000000
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        
        # Save final model
        model.save("./models/final_model")
        
        # Create equity curve and performance plot from callback history
        if len(callback.training_history['timesteps']) > 0:
            # Plot training equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(callback.training_history['timesteps'], callback.training_history['equity'])
            plt.title('Training Equity Curve')
            plt.xlabel('Timesteps')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.savefig('plots/training_equity_curve.png')
            plt.close()
            
            # Plot rolling reward
            if len(callback.training_history['rewards']) > 100:
                plt.figure(figsize=(12, 6))
                rewards = np.array(callback.training_history['rewards'])
                timesteps = np.array(callback.training_history['timesteps'])
                
                # Create 100-episode rolling average
                window = min(100, len(rewards) // 10)
                rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                rolling_timesteps = timesteps[window-1:]
                
                plt.plot(timesteps, rewards, 'b-', alpha=0.2)
                plt.plot(rolling_timesteps, rolling_rewards, 'r-', linewidth=2)
                plt.title('Episode Rewards During Training')
                plt.xlabel('Timesteps')
                plt.ylabel('Reward')
                plt.grid(True)
                plt.savefig('plots/training_rewards.png')
                plt.close()
    
    print("Generating test predictions...")
    
    # Generate weights for testing period
    weights = create_rl_weights(
        model=model,
        env=env,  # Pass unwrapped environment
        futures_data=futures_data,
        start_date="2024-01-01",
        end_date="2025-04-01"
    )
    
    # Clean the weights for Quantiacs standards
    weights = qnout.clean(weights, futures_data, "futures")
    
    # Calculate performance statistics
    stats = qnstats.calc_stat(futures_data, weights)
    print("Test Performance Statistics:")
    print(stats.to_pandas().tail())
    
    # Create equity curve
    performance = stats.to_pandas()["equity"]
    qngraph.make_plot_filled(performance.index, performance, name="RL Strategy PnL", type="log")
    
    # Display key statistics
    sharpe = stats.sel(field="sharpe_ratio").to_pandas().iloc[-1]
    max_dd = stats.sel(field="max_drawdown").to_pandas().iloc[-1]
    mean_ret = stats.sel(field="mean_return").to_pandas().iloc[-1]
    
    print(f"Final Sharpe Ratio: {sharpe}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
    print(f"Mean Annual Return: {mean_ret:.2%}")
    
    # Plot detailed performance
    returns = stats.sel(field="relative_return").to_pandas()
    plot_performance(returns, weights, title="RL S&P 500 Options Strategy")
    
    # Verify strategy meets requirements
    check_result = qnout.check(weights, futures_data, "futures")
    print("Strategy Check Result:", check_result)
    
    # Write weights for submission
    qnout.write(weights)
    
    print("RL model training and testing complete!")
    print("Results saved to 'plots/rl_model_performance.png'")
    print("Training progress saved to 'plots/training_progress.png'")
    print("Weights saved to 'fractions.nc.gz'")

if __name__ == "__main__":
    main() 