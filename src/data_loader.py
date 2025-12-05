"""
Data loading utilities for Bybit futures data with caching and visualization helpers.
"""

from __future__ import annotations

import os
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pybit.unified_trading import HTTP

from .config import (
    CACHE_DIR,
    DAYS_BACK,
    HOURS_BACK,
    MINUTES_BACK,
    INTERVAL,
    MAX_FETCH_BATCHES,
    MULTI_ASSET_CACHE,
    PLOT_TEMPLATE,
    SYMBOLS,
    get_lookback_timedelta,
)


class GlobalMarketDataset(Dataset):
    """
    Dataset that provides a synchronized global view of the market.
    
    Structure:
    - Loads multiple asset parquet files.
    - Aligns them on timestamp intersection (strict alignment).
    - Yields windows of shape (Sequence_Length, N_Assets, N_Features).
    """
    
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        sequence_length: int = 16,
        features: Optional[List[str]] = None,
    ):
        self.sequence_length = sequence_length
        self.features = features
        
        # Load and align data
        self.data, self.timestamps = self._load_and_align(file_paths)
        
        # Convert to tensor: (Time, N_Assets, N_Features)
        self.data_tensor = torch.FloatTensor(self.data)
        
    def _load_and_align(self, file_paths: List[Union[str, Path]]) -> tuple[np.ndarray, pd.DatetimeIndex]:
        dfs = []
        common_index = None
        
        print(f"[GLOBAL_LOADER] Loading {len(file_paths)} assets...")
        
        for path in file_paths:
            df = pd.read_parquet(path)
            
            # Ensure index is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)
            
            # Select features if specified
            if self.features:
                available_features = [f for f in self.features if f in df.columns]
                if len(available_features) < len(self.features):
                     # If missing features, we might need to handle it. For now, strict.
                     pass
                df = df[self.features]
            else:
                # Default to numeric columns if not specified
                df = df.select_dtypes(include=[np.number])
            
            dfs.append(df.sort_index())
            
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            raise ValueError("No overlapping timestamps found across assets!")
            
        print(f"[GLOBAL_LOADER] Aligned on {len(common_index)} timestamps.")
        
        # Reindex all dfs to common index and stack
        aligned_data = []
        for df in dfs:
            aligned_data.append(df.loc[common_index].values)
            
        # Stack to (Time, N_Assets, N_Features)
        # aligned_data is list of (Time, Features) arrays
        # We want (Time, Assets, Features)
        stacked = np.stack(aligned_data, axis=1)
        
        return stacked, common_index

    def __len__(self):
        return len(self.data_tensor) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Return window: (Sequence_Length, N_Assets, N_Features)
        return self.data_tensor[idx : idx + self.sequence_length]


def make_global_loader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    sequence_length: int = 16,
    features: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for the GlobalMarketDataset.
    
    Parameters
    ----------
    data_dir : str | Path
        Directory containing processed parquet files for each asset.
    batch_size : int
        Batch size.
    sequence_length : int
        Window size.
    features : List[str], optional
        List of feature columns to include.
    shuffle : bool
        Whether to shuffle the batches.
        
    Returns
    -------
    DataLoader
    """
    data_dir = Path(data_dir)
    # Assume all parquet files in dir are assets
    file_paths = sorted(list(data_dir.glob("*.parquet")))
    
    if not file_paths:
        raise ValueError(f"No parquet files found in {data_dir}")
        
    dataset = GlobalMarketDataset(
        file_paths=file_paths,
        sequence_length=sequence_length,
        features=features
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def validate_time_conversions(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate that time conversions are logically correct.
    
    Checks:
    1. Index is datetime
    2. No duplicate timestamps
    3. Timestamps are monotonically increasing
    4. No NaT (Not a Time) values
    5. Timestamps are in reasonable range (not in future, not before 2010)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    symbol : str
        Asset symbol (for error messages)
    
    Raises
    ------
    ValueError
        If any validation check fails
    """
    # Check 1: Index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: Index is not DatetimeIndex (type: {type(df.index)})")
    
    # Check 2: No duplicates
    if df.index.duplicated().any():
        n_dupes = df.index.duplicated().sum()
        raise ValueError(f"{symbol}: Found {n_dupes} duplicate timestamps")
    
    # Check 3: Monotonically increasing
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: Timestamps are not monotonically increasing")
    
    # Check 4: No NaT values
    if df.index.isna().any():
        n_nat = df.index.isna().sum()
        raise ValueError(f"{symbol}: Found {n_nat} NaT (Not a Time) values")
    
    # Check 5: Reasonable range
    now = datetime.utcnow()
    min_date = datetime(2010, 1, 1)  # Bitcoin genesis: 2009-01-03
    
    if df.index.min() < min_date:
        raise ValueError(f"{symbol}: Earliest timestamp ({df.index.min()}) is before 2010")
    
    if df.index.max() > now + timedelta(hours=1):  # Allow 1 hour clock skew
        raise ValueError(f"{symbol}: Latest timestamp ({df.index.max()}) is in the future")
    
    print(f"       [VALIDATION] ✓ Time conversions valid")
    print(f"       [VALIDATION] Range: {df.index.min()} to {df.index.max()}")
    print(f"       [VALIDATION] Samples: {len(df)}, Duplicates: 0, NaT: 0")


class MarketDataLoader:
    """
    Smart Bybit data loader that combines cached parquet storage with incremental API fetches.
    """

    def __init__(
        self,
        symbol: str = SYMBOLS[0],
        interval: str = INTERVAL,
        cache_dir: Path | str = CACHE_DIR,
        max_batches: int = MAX_FETCH_BATCHES,
        session: Optional[HTTP] = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{symbol.lower()}_{interval}m_cache.parquet"
        self.session = session or HTTP(testnet=False)
        self.max_batches = max_batches

    def get_data(
        self,
        days_back: Optional[int] = None,
        hours_back: Optional[int] = None,
        minutes_back: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV data for the requested window using local caches when possible.
        
        Parameters
        ----------
        days_back : int, optional
            Number of days to look back (default: DAYS_BACK from config)
        hours_back : int, optional
            Additional hours to look back (default: HOURS_BACK from config)
        minutes_back : int, optional
            Additional minutes to look back (default: MINUTES_BACK from config)
        start_time : datetime, optional
            Explicit start time (overrides days/hours/minutes)
        end_time : datetime, optional
            Explicit end time (default: now)
        force_refresh : bool
            Force fresh API fetch
        
        Returns
        -------
        pd.DataFrame
            OHLCV data with datetime index
        """
        end_time = end_time or datetime.utcnow()
        
        if start_time is None:
            # Use days/hours/minutes parameters
            lookback = get_lookback_timedelta(
                days=days_back,
                hours=hours_back,
                minutes=minutes_back
            )
            start_time = end_time - lookback
        
        # Validation: Ensure start_time < end_time
        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})")
        
        # Log the actual time window with duration breakdown
        duration = end_time - start_time
        duration_days = duration.days
        duration_hours = duration.seconds // 3600
        duration_minutes = (duration.seconds % 3600) // 60
        
        print(f"\n[DATA] Fetching {self.symbol} @ {self.interval}m")
        print(f"       Window: {start_time} to {end_time}")
        print(f"       Duration: {duration_days}d {duration_hours}h {duration_minutes}m")

        if force_refresh or not self.cache_file.exists():
            print("       Cache miss or forced refresh to fetching full range...")
            fresh = self._fetch_data(int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
            self._write_cache(fresh)
            return self._filter_range(fresh, start_time, end_time)

        cached_df = self._load_cache()
        
        # Check if cache is empty (corrupted or invalid)
        if cached_df.empty or len(cached_df) == 0:
            print("       Cache is empty, fetching full range...")
            fresh = self._fetch_data(int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
            self._write_cache(fresh)
            return self._filter_range(fresh, start_time, end_time)
        
        cache_start = cached_df.index[0].to_pydatetime()
        cache_end = cached_df.index[-1].to_pydatetime()

        print(f"       Cache range: {cache_start} to {cache_end}")

        frames = [cached_df]
        cache_updated = False

        if start_time < cache_start:
            print(f"       Cache missing head data before {cache_start}, fetching...")
            head = self._fetch_data(int(start_time.timestamp() * 1000), int(cache_start.timestamp() * 1000))
            if not head.empty:
                frames.append(head)
                cache_updated = True

        if end_time > cache_end:
            print(f"       Fetching incremental tail data since {cache_end}...")
            tail = self._fetch_data(int(cache_end.timestamp() * 1000), int(end_time.timestamp() * 1000))
            if not tail.empty:
                frames.append(tail)
                cache_updated = True

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        if cache_updated:
            self._write_cache(combined)
            print(f"       Cache updated ({len(combined)} total rows).")

        return self._filter_range(combined, start_time, end_time)

    def save_to_parquet(self, df: pd.DataFrame, path: Path | str) -> None:
        """Persist OHLCV data to a parquet file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_index().to_parquet(out_path)
        print(f"[DATA] Saved {len(df)} rows to {out_path}")

    def clear_cache(self) -> None:
        """Delete the cached parquet file, if it exists."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"[DATA] Cache cleared: {self.cache_file.name}")
        else:
            print("[DATA] No cache file found to delete.")

    def fetch_history(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Public wrapper around the internal fetcher."""
        return self._fetch_data(int(start_date.timestamp() * 1000), int(end_date.timestamp() * 1000))

    def fetch_all_assets(
        self,
        days_back: Optional[int] = None,
        hours_back: Optional[int] = None,
        minutes_back: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch and combine data for all configured assets.
        
        This method loops through all symbols in config.SYMBOLS, fetches data for each,
        adds asset_id and symbol columns, and concatenates into a single DataFrame.
        
        Parameters
        ----------
        days_back : int, optional
            Number of days of historical data to fetch for each asset
        hours_back : int, optional
            Additional hours of historical data to fetch
        minutes_back : int, optional
            Additional minutes of historical data to fetch
        force_refresh : bool
            If True, bypass cache and fetch fresh data
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with columns: [open, high, low, close, volume, asset_id, symbol]
            Index: timestamp (datetime)
        """
        cache_path = self.cache_dir / MULTI_ASSET_CACHE
        
        # Check if multi-asset cache exists and is valid
        if not force_refresh and cache_path.exists():
            print(f"\n[MULTI-ASSET] Loading from cache: {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            if "timestamp" in df_cached.columns:
                df_cached["timestamp"] = pd.to_datetime(df_cached["timestamp"])
                df_cached = df_cached.set_index("timestamp")
            else:
                df_cached.index = pd.to_datetime(df_cached.index)
            print(f"[MULTI-ASSET] Loaded {len(df_cached)} rows for {df_cached['asset_id'].nunique()} assets")
            return df_cached.sort_index()
        
        print(f"\n[MULTI-ASSET] Fetching data for {len(SYMBOLS)} assets...")
        frames = []
        
        for asset_id, symbol in enumerate(SYMBOLS):
            print(f"\n  [{asset_id + 1}/{len(SYMBOLS)}] Fetching {symbol}...")
            loader = MarketDataLoader(
                symbol=symbol,
                interval=self.interval,
                cache_dir=self.cache_dir,
                max_batches=self.max_batches,
                session=self.session,
            )
            
            try:
                df = loader.get_data(
                    days_back=days_back,
                    hours_back=hours_back,
                    minutes_back=minutes_back,
                    force_refresh=force_refresh
                )
                if df.empty:
                    print(f"      WARNING: No data retrieved for {symbol}, skipping...")
                    continue
                    
                df["asset_id"] = asset_id
                df["symbol"] = symbol
                frames.append(df)
                print(f"      [OK] Collected {len(df)} candles for {symbol}")
            except Exception as exc:
                print(f"      ERROR fetching {symbol}: {exc}")
                continue
        
        if not frames:
            raise ValueError("No data was successfully fetched for any asset!")
        
        # Combine all assets
        df_all = pd.concat(frames, axis=0).sort_index()
        
        # Save to multi-asset cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(cache_path)
        print(f"\n[MULTI-ASSET] Saved {len(df_all)} total rows to {cache_path}")
        print(f"[MULTI-ASSET] Assets: {', '.join(df_all['symbol'].unique())}")
        
        return df_all

    def _load_cache(self) -> pd.DataFrame:
        df = pd.read_parquet(self.cache_file)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _write_cache(self, df: pd.DataFrame) -> None:
        df.sort_index().to_parquet(self.cache_file)

    def _filter_range(self, df: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        mask = (df.index >= start_time) & (df.index <= end_time)
        filtered = df.loc[mask].copy()
        filtered = filtered[~filtered.index.duplicated(keep="first")]
        print(f"       Prepared {len(filtered)} candles.")
        return filtered

    def _fetch_data(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        print("       Fetching from Bybit in paginated batches...")
        all_data: list[list[str | float]] = []
        current_end = end_ms
        batch = 0

        while current_end > start_ms and batch < self.max_batches:
            try:
                response = self.session.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=self.interval,
                    end=current_end,
                    limit=1000,
                )
            except Exception as exc:
                safe_msg = str(exc).encode('ascii', 'replace').decode('ascii')
                print(f"       [ERROR] Fetch error: {safe_msg}")
                break

            if response["retCode"] != 0:
                print(f"       [ERROR] API Error: {response['retMsg']}")
                break

            klines = response["result"]["list"]
            if not klines:
                break

            for entry in klines:
                candle_ts = int(entry[0])
                if start_ms <= candle_ts <= end_ms:
                    all_data.append(entry)

            batch += 1
            print(f"         Batch {batch}: collected {len(all_data)} candles total.")
            oldest_ts = int(klines[-1][0]) - 1
            if oldest_ts <= start_ms:
                break
            current_end = oldest_ts
            time.sleep(0.1)

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df.drop(columns=["turnover"])
        df = df.sort_values("timestamp").set_index("timestamp")
        
        # VALIDATION: Check time conversions
        validate_time_conversions(df, self.symbol)
        
        return df


def visualize_data(df: pd.DataFrame, symbol: str = SYMBOLS[0]) -> Path:
    """
    Create an interactive candlestick + volume chart and open it in a browser.
    """
    working_df = df.copy()
    if "timestamp" not in working_df.columns:
        working_df = working_df.reset_index()

    working_df["SMA_20"] = working_df["close"].rolling(window=20).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} Price Chart", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=working_df["timestamp"],
            open=working_df["open"],
            high=working_df["high"],
            low=working_df["low"],
            close=working_df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=working_df["timestamp"],
            y=working_df["SMA_20"],
            mode="lines",
            name="SMA 20",
            line=dict(color="#FFA726", width=2),
        ),
        row=1,
        col=1,
    )

    colors = ["#ef5350" if close < open_ else "#26a69a" for close, open_ in zip(working_df["close"], working_df["open"])]
    fig.add_trace(
        go.Bar(
            x=working_df["timestamp"],
            y=working_df["volume"],
            name="Volume",
            marker_color=colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} Futures - Interactive Chart",
        yaxis_title="Price (USDT)",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        template=PLOT_TEMPLATE,
        height=800,
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)

    filename = Path(f"chart_{symbol}.html")
    fig.write_html(filename)
    print(f"[PLOT] Chart saved to {filename}")

    filepath = Path(os.path.abspath(filename))
    webbrowser.open(f"file://{filepath}")
    print("[PLOT] Opening chart in browser...")
    return filepath
