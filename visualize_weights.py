import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_weights(file_path='fractions.nc.gz'):
    """Load strategy weights from a .nc.gz file"""
    print(f"Loading weights from {file_path}...")
    weights = xr.open_dataarray(file_path)
    print(f"Loaded weights with shape: {weights.shape}")
    print(f"Time range: {weights.time.values[0]} to {weights.time.values[-1]}")
    print(f"Number of assets: {len(weights.asset.values)}")
    return weights

def plot_asset_allocation_heatmap(weights, top_n=20, save_path='plots/asset_allocation_heatmap.png'):
    """Create a heatmap of top asset allocations over time"""
    # Identify top assets by average absolute weight
    abs_weights = abs(weights)
    mean_weights = abs_weights.mean(dim='time')
    top_assets = mean_weights.sortby(mean_weights, ascending=False).asset.values[:top_n]
    
    # Filter for top assets
    top_weights = weights.sel(asset=top_assets)
    
    # Convert to pandas and resample to monthly for better visualization
    weights_df = top_weights.to_pandas()
    
    # Create pivot table with assets as columns and time as index
    # Resample to monthly average
    monthly_weights = weights_df.resample('M').mean()
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(monthly_weights.T, cmap='RdBu_r', center=0, 
                vmin=-0.1, vmax=0.1, 
                xticklabels=monthly_weights.index.strftime('%Y-%m'),
                yticklabels=monthly_weights.columns)
    plt.title(f'Monthly Average Position Weights for Top {top_n} Assets')
    plt.ylabel('Asset')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved asset allocation heatmap to {save_path}")
    plt.close()

def plot_top_positions_over_time(weights, date_interval='Y', 
                                save_path='plots/top_positions_over_time.png'):
    """Plot top long and short positions at regular intervals"""
    # Get dates at regular intervals
    unique_dates = pd.to_datetime(weights.time.values)
    
    if date_interval == 'Y':
        # Select one date per year
        selected_dates = []
        current_year = None
        for date in unique_dates:
            if current_year != date.year:
                selected_dates.append(date)
                current_year = date.year
    else:
        # Use all dates
        selected_dates = unique_dates
    
    # Limit to a reasonable number of dates for visualization
    if len(selected_dates) > 10:
        step = len(selected_dates) // 10
        selected_dates = selected_dates[::step]
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(len(selected_dates), 2, figsize=(15, len(selected_dates) * 3))
    
    for i, date in enumerate(selected_dates):
        # Get weights for this date
        date_weights = weights.sel(time=date, method='nearest')
        
        # Sort for top long and short positions
        long_positions = date_weights.sortby(date_weights, ascending=False)
        short_positions = date_weights.sortby(date_weights, ascending=True)
        
        # Convert to pandas Series for bar plotting
        # Plot top 5 long positions
        ax_long = axes[i, 0] if len(selected_dates) > 1 else axes[0]
        top_long = long_positions.isel(asset=slice(0, 5))
        top_long_series = top_long.to_pandas()
        top_long_series.plot.bar(ax=ax_long, color='green')
        ax_long.set_title(f'Top 5 Long Positions - {pd.to_datetime(date).strftime("%Y-%m-%d")}')
        ax_long.set_ylabel('Weight')
        ax_long.set_ylim(0, 0.15)  # Adjust as needed
        
        # Plot top 5 short positions
        ax_short = axes[i, 1] if len(selected_dates) > 1 else axes[1]
        top_short = short_positions.isel(asset=slice(0, 5))
        top_short_series = top_short.to_pandas()
        top_short_series.plot.bar(ax=ax_short, color='red')
        ax_short.set_title(f'Top 5 Short Positions - {pd.to_datetime(date).strftime("%Y-%m-%d")}')
        ax_short.set_ylabel('Weight')
        ax_short.set_ylim(-0.15, 0)  # Adjust as needed
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved top positions plot to {save_path}")
    plt.close()

def plot_weight_distribution(weights, save_path='plots/weight_distribution.png'):
    """Plot the distribution of weights"""
    plt.figure(figsize=(12, 6))
    
    # Flatten weights and remove NaN values
    flat_weights = weights.values.flatten()
    flat_weights = flat_weights[~np.isnan(flat_weights)]
    
    # Plot histogram with KDE
    sns.histplot(flat_weights, kde=True, bins=50)
    plt.title('Distribution of Position Weights')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    
    # Add vertical lines at mean and median
    plt.axvline(np.mean(flat_weights), color='r', linestyle='--', label=f'Mean: {np.mean(flat_weights):.4f}')
    plt.axvline(np.median(flat_weights), color='g', linestyle='--', label=f'Median: {np.median(flat_weights):.4f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved weight distribution plot to {save_path}")
    plt.close()

def plot_exposure_over_time(weights, save_path='plots/exposure_over_time.png'):
    """Plot net exposure and gross exposure over time"""
    # Calculate exposures
    net_exposure = weights.sum(dim='asset')
    gross_exposure = abs(weights).sum(dim='asset')
    long_exposure = weights.where(weights > 0, 0).sum(dim='asset')
    short_exposure = abs(weights.where(weights < 0, 0).sum(dim='asset'))
    
    # Convert to pandas for plotting
    net_df = net_exposure.to_pandas()
    gross_df = gross_exposure.to_pandas()
    long_df = long_exposure.to_pandas()
    short_df = short_exposure.to_pandas()
    
    # Resample to weekly for smoother visualization
    net_weekly = net_df.resample('W').mean()
    gross_weekly = gross_df.resample('W').mean()
    long_weekly = long_df.resample('W').mean()
    short_weekly = short_df.resample('W').mean()
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.plot(net_weekly.index, net_weekly.values, 'k-', label='Net Exposure')
    plt.plot(gross_weekly.index, gross_weekly.values, 'b-', label='Gross Exposure')
    plt.plot(long_weekly.index, long_weekly.values, 'g-', label='Long Exposure')
    plt.plot(short_weekly.index, short_weekly.values, 'r-', label='Short Exposure')
    
    plt.title('Portfolio Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved exposure plot to {save_path}")
    plt.close()

def plot_turnover(weights, lookback=20, save_path='plots/turnover.png'):
    """Plot portfolio turnover (sum of absolute weight changes)"""
    # Calculate weight changes day-to-day
    weight_changes = abs(weights - weights.shift(time=1))
    turnover = weight_changes.sum(dim='asset')
    
    # Calculate rolling average turnover
    rolling_turnover = turnover.rolling(time=lookback).mean()
    
    # Convert to pandas for plotting
    turnover_df = turnover.to_pandas()
    rolling_turnover_df = rolling_turnover.to_pandas()
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(turnover_df.index, turnover_df.values, 'b-', alpha=0.3, label='Daily Turnover')
    plt.plot(rolling_turnover_df.index, rolling_turnover_df.values, 'r-', label=f'{lookback}-Day Average Turnover')
    
    plt.title('Portfolio Turnover Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turnover (Sum of Absolute Weight Changes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved turnover plot to {save_path}")
    plt.close()

def main():
    # Load weights
    weights = load_weights()
    
    # Generate visualizations
    plot_asset_allocation_heatmap(weights)
    plot_top_positions_over_time(weights)
    plot_weight_distribution(weights)
    plot_exposure_over_time(weights)
    plot_turnover(weights)
    
    print("All visualizations completed and saved to the 'plots' directory")

if __name__ == "__main__":
    main() 