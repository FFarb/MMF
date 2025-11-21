"""
Signal Factory Module
=====================
Automated feature engineering pipeline for generating massive alpha signal sets.
Generates 1000+ technical features from OHLCV data using pandas_ta.

Author: Quant Data Engineer
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from data_manager import BybitDataManager
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SignalFactory:
    """
    A factory class to generate technical signals and features from OHLCV data.
    Designed for high-performance and memory efficiency.
    """
    
    def __init__(self):
        self.windows = [3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
        self.lags = [1, 2, 3, 5, 8, 13]
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to generate all signals.
        
        Args:
            df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            pd.DataFrame: DataFrame with original data + engineered features
        """
        print(f"[INFO] Starting Signal Factory on {len(df)} rows...")
        
        # Ensure float32 for memory efficiency
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
                
        # Copy to avoid setting on copy warnings
        df = df.copy()
        
        # ====================================================================
        # STEP A: PRICE TRANSFORMS
        # ====================================================================
        print("  [STEP A] Generating Price Transforms...")
        
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).astype(np.float32)
        
        # Log Range (Volatility proxy)
        df['log_range'] = np.log(df['high'] / df['low']).astype(np.float32)
        
        # Body-to-Shadow Ratios
        body_size = np.abs(df['close'] - df['open'])
        shadow_size = (df['high'] - df['low']) - body_size
        df['body_shadow_ratio'] = (body_size / (shadow_size + 1e-9)).astype(np.float32)
        
        # Upper/Lower Shadow ratios
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        df['upper_shadow_ratio'] = (upper_shadow / (body_size + 1e-9)).astype(np.float32)
        df['lower_shadow_ratio'] = (lower_shadow / (body_size + 1e-9)).astype(np.float32)
        
        # ====================================================================
        # STEP B: PARAMETRIC INDICATORS (LOOP OVER WINDOWS)
        # ====================================================================
        print(f"  [STEP B] Generating Parametric Indicators (Windows: {self.windows})...")
"""
Signal Factory Module
=====================
Automated feature engineering pipeline for generating massive alpha signal sets.
Generates 1000+ technical features from OHLCV data using pandas_ta.

Author: Quant Data Engineer
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from data_manager import BybitDataManager
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SignalFactory:
    """
    A factory class to generate technical signals and features from OHLCV data.
    Designed for high-performance and memory efficiency.
    """
    
    def __init__(self):
        self.windows = [3, 5, 8, 13, 21, 34, 55, 89, 144, 200]
        self.lags = [1, 2, 3, 5, 8, 13]
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to generate all signals.
        
        Args:
            df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            pd.DataFrame: DataFrame with original data + engineered features
        """
        print(f"[INFO] Starting Signal Factory on {len(df)} rows...")
        
        # Ensure float32 for memory efficiency
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
                
        # Copy to avoid setting on copy warnings
        df = df.copy()
        
        # ====================================================================
        # STEP A: PRICE TRANSFORMS
        # ====================================================================
        print("  [STEP A] Generating Price Transforms...")
        
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).astype(np.float32)
        
        # Log Range (Volatility proxy)
        df['log_range'] = np.log(df['high'] / df['low']).astype(np.float32)
        
        # Body-to-Shadow Ratios
        body_size = np.abs(df['close'] - df['open'])
        shadow_size = (df['high'] - df['low']) - body_size
        df['body_shadow_ratio'] = (body_size / (shadow_size + 1e-9)).astype(np.float32)
        
        # Upper/Lower Shadow ratios
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        df['upper_shadow_ratio'] = (upper_shadow / (body_size + 1e-9)).astype(np.float32)
        df['lower_shadow_ratio'] = (lower_shadow / (body_size + 1e-9)).astype(np.float32)
        
        # ====================================================================
        # STEP B: PARAMETRIC INDICATORS (LOOP OVER WINDOWS)
        # ====================================================================
        print(f"  [STEP B] Generating Parametric Indicators (Windows: {self.windows})...")
        
        for w in self.windows:
            # Momentum
            df[f'RSI_{w}'] = ta.rsi(df['close'], length=w).astype(np.float32)
            df[f'ROC_{w}'] = ta.roc(df['close'], length=w).astype(np.float32)
            df[f'CCI_{w}'] = ta.cci(df['high'], df['low'], df['close'], length=w).astype(np.float32)
            df[f'WILLR_{w}'] = ta.willr(df['high'], df['low'], df['close'], length=w).astype(np.float32)
            
            # Volatility
            df[f'ATR_{w}'] = ta.atr(df['high'], df['low'], df['close'], length=w).astype(np.float32)
            df[f'NATR_{w}'] = ta.natr(df['high'], df['low'], df['close'], length=w).astype(np.float32)
            
            # Trend
            adx = ta.adx(df['high'], df['low'], df['close'], length=w)
            if adx is not None:
                df[f'ADX_{w}'] = adx[f'ADX_{w}'].astype(np.float32)
                df[f'DMP_{w}'] = adx[f'DMP_{w}'].astype(np.float32)
                df[f'DMN_{w}'] = adx[f'DMN_{w}'].astype(np.float32)
            
            # Averages (Distances)
            sma = ta.sma(df['close'], length=w)
            ema = ta.ema(df['close'], length=w)
            df[f'dist_SMA_{w}'] = ((df['close'] - sma) / sma).astype(np.float32)
            df[f'dist_EMA_{w}'] = ((df['close'] - ema) / ema).astype(np.float32)
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=w, std=2)
            if bb is not None:
                df[f'BB_pctB_{w}'] = bb.iloc[:, 4].astype(np.float32)
                df[f'BB_width_{w}'] = bb.iloc[:, 3].astype(np.float32)
                
            # Volume
            if w < 50: # MFI requires volume, usually shorter windows
                df[f'MFI_{w}'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=w).astype(np.float32)

        # Additional Indicators (Fixed windows or standard)
        df['OBV'] = ta.obv(df['close'], df['volume']).astype(np.float32)
        df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20).astype(np.float32)
        
        macds = ta.macd(df['close'])
        if macds is not None:
            df['MACD'] = macds.iloc[:, 0].astype(np.float32)
            df['MACD_hist'] = macds.iloc[:, 1].astype(np.float32)
            df['MACD_signal'] = macds.iloc[:, 2].astype(np.float32)

        # ====================================================================
        # STEP C: MATH / STATISTICAL
        # ====================================================================
        print("  [STEP C] Generating Statistical Features...")
        
        stat_windows = [20, 50, 100]
        for w in stat_windows:
            # Rolling Skew & Kurtosis
            df[f'skew_{w}'] = df['close'].rolling(window=w).skew().astype(np.float32)
            df[f'kurt_{w}'] = df['close'].rolling(window=w).kurt().astype(np.float32)
            
            # Rolling Z-Score
            rolling_mean = df['close'].rolling(window=w).mean()
            rolling_std = df['close'].rolling(window=w).std()
            df[f'zscore_{w}'] = ((df['close'] - rolling_mean) / rolling_std).astype(np.float32)
            
            # Slope (Linear Regression Angle)
            # Using numpy polyfit on rolling windows is slow, using simple rise/run proxy or ta.slope
            df[f'slope_{w}'] = ta.slope(df['close'], length=w).astype(np.float32)
            
        # ====================================================================
        # STEP D: LAGGING (TEMPORAL FEATURES)
        # ====================================================================
        print("  [STEP D] Generating Lagged Features (Massive Scale)...")
        
        # Identify all numeric feature columns (excluding original OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"    Base features to lag: {len(feature_cols)}")
        
        # Lag ALL features to maximize count
        for feature in feature_cols:
            for lag in self.lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag).astype(np.float32)
                
        # ====================================================================
        # STEP E: CLEANUP & MEMORY OPTIMIZATION
        # ====================================================================
        print("  [STEP E] Cleanup...")
        
        # Drop initial rows with NaNs (due to largest window)
        max_window = max(self.windows)
        df = df.iloc[max_window:].copy()
        
        # Drop any remaining NaNs
        initial_len = len(df)
        df = df.dropna()
        if len(df) < initial_len:
            print(f"    Dropped {initial_len - len(df)} additional rows with NaNs")
            
        # Final float32 check
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)
            
        print(f"[SUCCESS] Signal Generation Complete. Final Shape: {df.shape}")
        return df

if __name__ == "__main__":
    # 1. Load Data
    print("="*70)
    print("SIGNAL FACTORY EXECUTION")
    print("="*70)
    
    # Check if we have the specific file, otherwise use DataManager
    try:
        if pd.io.common.file_exists('btc_futures.parquet'):
            print("Loading local btc_futures.parquet...")
            df = pd.read_parquet('btc_futures.parquet')
        else:
            print("Fetching data via DataManager...")
            manager = BybitDataManager(symbol="BTCUSDT", interval="15")
            # Fetch ample data to ensure we have enough after dropping NaNs
            df = manager.get_data(days_back=180) 
            # Save for future use
            df.to_parquet('btc_futures.parquet')
            print("Saved to btc_futures.parquet")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()
        
    # 2. Run Factory
    factory = SignalFactory()
    df_features = factory.generate_signals(df)
    
    # 3. Validation
    print("\n[INFO] Feature Verification:")
    print(f"Total Features Generated: {len(df_features.columns)}")
    
    print("\nFirst 10 Features:")
    print(df_features.columns[:10].tolist())
    
    print("\nLast 10 Features:")
    print(df_features.columns[-10:].tolist())
    
    # 4. Save
    output_file = 'btc_1000_features.parquet'
    df_features.to_parquet(output_file)
    print(f"\n[SUCCESS] Saved {len(df_features)} rows to {output_file}")
    print("="*70)
