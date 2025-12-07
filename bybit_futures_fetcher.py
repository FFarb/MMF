"""
Bybit Futures Data Fetcher and Visualizer
------------------------------------------
Fetches historical futures data from Bybit and creates interactive visualizations.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import time
import webbrowser
import os


class BybitFuturesLoader:
    """
    Class for fetching historical futures data from Bybit V5 API.
    """
    
    def __init__(self):
        """Initialize the Bybit HTTP session."""
        self.session = HTTP(testnet=False)
    
    def fetch_history(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical kline data from Bybit with automatic pagination.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '60' for 1 hour, '240' for 4 hours)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        print(f"Fetching {symbol} data from {start_date} to {end_date}...")
        
        # Convert dates to millisecond timestamps
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_data = []
        current_end = end_ms
        
        # Bybit returns data in reverse chronological order, so we work backwards
        while current_end > start_ms:
            try:
                response = self.session.get_kline(
                    category="linear",  # USDT Perpetual Futures
                    symbol=symbol,
                    interval=interval,
                    end=current_end,
                    limit=1000  # Maximum allowed per request
                )
                
                if response['retCode'] != 0:
                    print(f"API Error: {response['retMsg']}")
                    break
                
                klines = response['result']['list']
                
                if not klines:
                    break
                
                all_data.extend(klines)
                
                # Get the timestamp of the oldest candle in this batch
                oldest_timestamp = int(klines[-1][0])
                
                # Break if we've reached the start date
                if oldest_timestamp <= start_ms:
                    break
                
                # Update current_end for next iteration
                current_end = oldest_timestamp
                
                print(f"Fetched {len(klines)} candles... Total: {len(all_data)}")
                
                # Be respectful to API rate limits
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = str(e)
                # Handle encoding issues by converting to ascii
                safe_error = error_msg.encode('ascii', 'replace').decode('ascii')
                print(f"Error fetching data: {safe_error}")
                
                # Check if it's a rate limit or access error
                if "403" in error_msg or "rate limit" in error_msg.lower():
                    print("\n" + "="*60)
                    print("API ACCESS WARNING")
                    print("="*60)
                    print("Bybit API returned an access error (403).")
                    print("This could be due to:")
                    print("  1. IP rate limiting")
                    print("  2. Geographic restrictions (US IP addresses are blocked)")
                    print("  3. Network/firewall issues")
                    print("\nPossible solutions:")
                    print("  - Use a VPN to access from a non-US location")
                    print("  - Wait a few minutes and try again")
                    print("  - Check your internet connection")
                    print("="*60)
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        # Convert price and volume columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Drop the turnover column as it's not needed
        df = df.drop('turnover', axis=1)
        
        # Filter to exact date range and sort by timestamp
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Successfully fetched {len(df)} candles")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (e.g., 'btc_futures.parquet')
        """
        df.to_parquet(filename, index=False)
        print(f"Data saved to {filename}")


def visualize_data(df: pd.DataFrame, symbol: str):
    """
    Create an interactive financial chart with candlesticks, volume, and SMA.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for the chart title
    """
    # Calculate 20-period Simple Moving Average
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Create subplots: candlestick on top, volume on bottom
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} Price Chart', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add 20-period SMA
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#FFA726', width=2)
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#ef5350' if close < open else '#26a69a' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Futures - Interactive Chart',
        yaxis_title='Price (USDT)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800,
        hovermode='x unified'
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Save to HTML
    filename = f'chart_{symbol}.html'
    fig.write_html(filename)
    print(f"Chart saved to {filename}")
    
    # Open in browser
    filepath = os.path.abspath(filename)
    webbrowser.open(f'file://{filepath}')
    print("Opening chart in browser...")


if __name__ == "__main__":
    # Initialize the loader
    loader = BybitFuturesLoader()
    
    # Define parameters
    symbol = "BTCUSDT"
    interval = "60"  # 1 hour
    
    # Calculate date range (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    print(f"\n{'='*60}")
    print(f"Bybit Futures Data Fetcher")
    print(f"{'='*60}\n")
    
    # Fetch data
    df = loader.fetch_history(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save to parquet
    parquet_file = "btc_futures.parquet"
    loader.save_to_parquet(df, parquet_file)
    
    print(f"\n{'='*60}")
    print(f"Data Summary")
    print(f"{'='*60}")
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"{'='*60}\n")
    
    # Visualize
    visualize_data(df, symbol)
