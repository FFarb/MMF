"""
Optimized Data Manager for Bybit Futures
Implements smart caching to minimize API calls and speed up data fetching
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from pathlib import Path
import time

class BybitDataManager:
    """
    Smart data manager with caching capabilities
    - Caches data locally in parquet format
    - Only fetches new data on updates (incremental fetch)
    - Dramatically reduces fetch time for subsequent runs
    """
    
    def __init__(self, symbol="BTCUSDT", interval="60", cache_dir="."):
        self.symbol = symbol
        self.interval = interval
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{symbol.lower()}_{interval}m_cache.parquet"
        self.session = HTTP(testnet=False)
        
    def get_data(self, days_back=90, force_refresh=False):
        """
        Get historical kline data with smart caching
        
        Args:
            days_back: Number of days of historical data
            force_refresh: If True, ignore cache and fetch all data fresh
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        print(f"\n[DATA MANAGER] Fetching {self.symbol} data")
        print(f"  Target: Last {days_back} days of {self.interval}min data")
        
        # Calculate target time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Check if cache exists and is usable
        if self.cache_file.exists() and not force_refresh:
            print(f"  [OK] Cache found: {self.cache_file.name}")
            cached_df = pd.read_parquet(self.cache_file)
            
            # Check cache freshness
            cache_end = cached_df.index[-1].to_pydatetime()
            cache_start = cached_df.index[0].to_pydatetime()
            
            print(f"  Cache range: {cache_start} to {cache_end}")
            
            # If cache is recent enough, just fetch new data
            if cache_end > start_time:
                time_diff = (end_time - cache_end).total_seconds() / 3600
                
                if time_diff < 1:  # Less than 1 hour old
                    print(f"  [OK] Cache is fresh ({time_diff:.1f}h old), using cached data")
                    return self._filter_and_prepare(cached_df, start_time, end_time)
                else:
                    print(f"  [UPDATE] Fetching only NEW data since {cache_end} ({time_diff:.1f}h)")
                    new_data = self._fetch_data(
                        int(cache_end.timestamp() * 1000), 
                        int(end_time.timestamp() * 1000)
                    )
                    
                    if not new_data.empty:
                        # Combine old and new data
                        combined_df = pd.concat([cached_df, new_data]).drop_duplicates()
                        combined_df = combined_df.sort_index()
                        
                        # Save updated cache
                        combined_df.to_parquet(self.cache_file)
                        print(f"  [OK] Updated cache with {len(new_data)} new candles")
                        
                        return self._filter_and_prepare(combined_df, start_time, end_time)
                    else:
                        print(f"  [WARNING] No new data available")
                        return self._filter_and_prepare(cached_df, start_time, end_time)
        
        # No cache or force refresh - fetch all data
        print(f"  [FETCH] Fetching ALL data from scratch...")
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        df = self._fetch_data(start_ms, end_ms)
        
        # Save to cache
        df.to_parquet(self.cache_file)
        print(f"  [OK] Saved {len(df)} candles to cache")
        
        return df
    
    def _fetch_data(self, start_ms, end_ms):
        """Fetch data from Bybit API with optimized batch size"""
        all_data = []
        current_start = start_ms
        batch_count = 0
        MAX_BATCHES = 10  # Safety: 10 batches = 10,000 candles max
        
        print(f"  Fetching in batches (max 1000/request)...")
        start_fetch_time = time.time()
        
        while current_start < end_ms and batch_count < MAX_BATCHES:
            try:
                response = self.session.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=self.interval,
                    end=end_ms,  # Add end time to limit range
                    limit=1000
                )
                
                if response['retCode'] == 0:
                    klines = response['result']['list']
                    if not klines:
                        break
                    
                    # Filter only data in our range
                    for kline in klines:
                        kline_ts = int(kline[0])
                        if start_ms <= kline_ts <= end_ms:
                            all_data.append(kline)
                    
                    batch_count += 1
                    
                    # Progress indicator  
                    print(f"    Batch {batch_count}: {len(all_data)} candles collected")
                    
                    # Move to next batch (go backwards in time)
                    oldest_ts = int(klines[-1][0])
                    if oldest_ts <= start_ms:
                        break
                    current_start = oldest_ts - 1
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                else:
                    print(f"  [ERROR] API Error: {response['retMsg']}")
                    break
                    
            except Exception as e:
                print(f"  [ERROR] Fetch error: {e}")
                break
        
        fetch_duration = time.time() - start_fetch_time
        print(f"  [OK] Fetched {len(all_data)} candles in {fetch_duration:.2f}s ({batch_count} batches)")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Sort and set index
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _filter_and_prepare(self, df, start_time, end_time):
        """Filter DataFrame to target date range"""
        # Filter to requested time range
        mask = (df.index >= start_time) & (df.index <= end_time)
        filtered_df = df[mask].copy()
        
        # Remove duplicate timestamps (keep first occurrence)
        filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]
        
        print(f"  [OK] Prepared {len(filtered_df)} candles")
        print(f"  Date range: {filtered_df.index[0]} to {filtered_df.index[-1]}")
        
        return filtered_df
    
    def clear_cache(self):
        """Delete cache file"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"  [OK] Cache cleared: {self.cache_file.name}")
        else:
            print(f"  [WARNING] No cache to clear")


if __name__ == "__main__":
    # Test the data manager
    print("="*70)
    print("TESTING DATA MANAGER")
    print("="*70)
    
    manager = BybitDataManager()
    
    # First run - will fetch all data
    print("\n--- TEST 1: Initial fetch (should be slow) ---")
    df1 = manager.get_data(days_back=90)
    print(f"\nResult: {len(df1)} rows")
    
    # Second run - should use cache (fast!)
    print("\n--- TEST 2: Using cache (should be instant) ---")
    df2 = manager.get_data(days_back=90)
    print(f"\nResult: {len(df2)} rows")
    
    # Force refresh
    print("\n--- TEST 3: Force refresh ---")
    df3 = manager.get_data(days_back=90, force_refresh=True)
    print(f"\nResult: {len(df3)} rows")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
