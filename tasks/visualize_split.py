import matplotlib.pyplot as plt
import pandas as pd
from data_utils import WeatherDataset

def plot_data_distribution():
    print("Initializing Dataset...")
    # 1. Load Data using your existing utility
    ds = WeatherDataset()
    df = ds.df
    
    # 2. Calculate the Split Index (80%)
    # We do this exactly as done in get_train_test_split
    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)
    
    # Get the date where the split happens for visualization
    split_date = df['time'].iloc[split_idx]
    
    print(f"Total Rows: {len(df)}")
    print(f"Training Rows: {split_idx}")
    print(f"Testing Rows: {len(df) - split_idx}")
    print(f"Split Date: {split_date.date()}")

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Plot 1: Temperature ---
    col_temp = ds.col_temp
    
    # Plot Training part
    ax1.plot(df['time'][:split_idx], df[col_temp][:split_idx], 
             label='Training Data (80%)', color='#1f77b4', linewidth=1)
    
    # Plot Testing part
    ax1.plot(df['time'][split_idx:], df[col_temp][split_idx:], 
             label='Testing Data (20%)', color='#ff7f0e', linewidth=1)
    
    # Draw Split Line
    ax1.axvline(x=split_date, color='black', linestyle='--', linewidth=1.5, label='Split Point')
    
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Actual Data Split: {col_temp}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: ET0 ---
    col_et0 = ds.col_et0
    
    # Plot Training part
    ax2.plot(df['time'][:split_idx], df[col_et0][:split_idx], 
             label='Training Data (80%)', color='#2ca02c', linewidth=1)
    
    # Plot Testing part
    ax2.plot(df['time'][split_idx:], df[col_et0][split_idx:], 
             label='Testing Data (20%)', color='#d62728', linewidth=1)
    
    # Draw Split Line
    ax2.axvline(x=split_date, color='black', linestyle='--', linewidth=1.5, label='Split Point')
    
    ax2.set_ylabel('Evapotranspiration (mm)')
    ax2.set_title(f'Actual Data Split: {col_et0}')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    print("Displaying plot...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_data_distribution()