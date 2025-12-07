"""
Demo script with sample data for Bybit Futures Visualizer
----------------------------------------------------------
This demonstrates the visualization capabilities using synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bybit_futures_fetcher import visualize_data

# Generate sample OHLCV data
def generate_sample_data(days=90):
    """Generate realistic sample OHLCV data"""
    
    # Create timestamps for last 90 days, hourly
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    # Generate price data with realistic movement
    np.random.seed(42)
    n_periods = len(timestamps)
    
    # Start at realistic BTC price
    base_price = 65000
    returns = np.random.normal(0.0001, 0.01, n_periods)  # Small mean, realistic volatility
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC from the price series
    data = []
    for i, ts in enumerate(timestamps):
        close_price = price_series[i]
        
        # Create realistic OHLC based on close
        daily_range = close_price * np.random.uniform(0.005, 0.025)
        open_price = close_price + np.random.uniform(-daily_range/2, daily_range/2)
        high_price = max(open_price, close_price) + np.random.uniform(0, daily_range/2)
        low_price = min(open_price, close_price) - np.random.uniform(0, daily_range/2)
        
        # Volume varies realistically
        volume = np.random.uniform(5000, 25000)
        
        data.append({
            'timestamp': ts,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Bybit Futures Demo - Sample Data Visualization")
    print("="*60)
    print("\nGenerating sample BTCUSDT data...")
    
    # Generate sample data
    df = generate_sample_data(days=90)
    
    # Save to parquet
    parquet_file = "btc_futures_demo.parquet"
    df.to_parquet(parquet_file, index=False)
    print(f"Sample data saved to {parquet_file}")
    
    print(f"\n{'='*60}")
    print(f"Data Summary")
    print(f"{'='*60}")
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"{'='*60}\n")
    
    # Visualize
    print("Creating interactive chart...")
    visualize_data(df, "BTCUSDT_DEMO")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
