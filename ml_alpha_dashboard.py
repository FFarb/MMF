"""
ML Alpha Research Dashboard
============================
A comprehensive tool for ML-based alpha research on crypto futures.
Allows feature engineering, model training, and feature importance analysis.

Author: Senior Quant Developer
"""

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from data_manager import BybitDataManager
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ML Alpha Research Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3241;
    }
    h1 {
        color: #00d4ff;
        font-weight: 700;
    }
    h2 {
        color: #00b4d8;
    }
    h3 {
        color: #90e0ef;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.title("🧠 ML Alpha Research Dashboard")
st.markdown("**Quantitative Feature Engineering & Model Training Platform**")
st.markdown("---")

# ============================================================================
# FEATURE ENGINEERING - "THE ALPHA FACTORY"
# ============================================================================
@st.cache_data
def add_features(df):
    """
    The Alpha Factory: Generate 30+ technical features
    
    Categories:
    - Momentum: RSI, MACD, ROC, AO, STOCH
    - Volatility: ATR, BB Width, Donchian Width
    - Volume: OBV, CMF, Volume Ratio
    - Trend: ADX, SMA Slope
    - Lagged: Log Returns (t-1, t-2, t-3, t-5)
    - Time: Hour, Day of Week
    """
    df = df.copy()
    
    # ========== MOMENTUM INDICATORS ==========
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_diff'] = macd['MACDh_12_26_9']
    
    # Rate of Change
    df['ROC_10'] = ta.roc(df['close'], length=10)
    
    # Awesome Oscillator
    df['AO'] = ta.ao(df['high'], df['low'])
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['STOCH_k'] = stoch['STOCHk_14_3_3']
    df['STOCH_d'] = stoch['STOCHd_14_3_3']
    
    # ========== VOLATILITY INDICATORS ==========
    # ATR
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Bollinger Bands Width
    bbands = ta.bbands(df['close'], length=20, std=2)
    # Use iloc to avoid column name issues (Lower=0, Mid=1, Upper=2)
    df['BB_width'] = (bbands.iloc[:, 2] - bbands.iloc[:, 0]) / bbands.iloc[:, 1]
    
    # Donchian Channel Width
    donchian = ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
    df['Donchian_width'] = (donchian['DCU_20_20'] - donchian['DCL_20_20']) / df['close']
    
    # ========== VOLUME INDICATORS ==========
    # On-Balance Volume
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['OBV_norm'] = df['OBV'] / df['OBV'].rolling(50).mean()
    
    # Chaikin Money Flow
    df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
    
    # Volume Ratio
    df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_SMA_20']
    
    # ========== TREND INDICATORS ==========
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    
    # SMA Slope (angle of trend)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_slope'] = df['SMA_50'].diff(5) / df['SMA_50']
    
    # ========== LAGGED FEATURES ==========
    # Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_t1'] = df['log_return'].shift(1)
    df['log_return_t2'] = df['log_return'].shift(2)
    df['log_return_t3'] = df['log_return'].shift(3)
    df['log_return_t5'] = df['log_return'].shift(5)
    
    # Rolling Statistics
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['return_mean_10'] = df['log_return'].rolling(10).mean()
    
    # ========== TIME FEATURES ==========
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # ========== PRICE ACTION ==========
    # High-Low Range
    df['HL_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Close position in range
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # ========== ADDITIONAL MOMENTUM ==========
    # Williams %R
    df['WILLR'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    
    # Commodity Channel Index
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # Money Flow Index
    df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    # ========== CLEAN UP ==========
    # Drop rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    st.info(f"✅ Generated **{len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])}** features. Dropped {dropped_rows} rows with NaN.")
    
    return df


# ============================================================================
# TARGET CREATION
# ============================================================================
def create_target(df, horizon, threshold_pct):
    """
    Create binary target based on future price movement
    
    Target = 1 if price moves up by threshold_pct% within horizon periods
    Target = 0 otherwise
    """
    df = df.copy()
    
    # Calculate future return
    df['future_close'] = df['close'].shift(-horizon)
    df['future_return_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100
    
    # Binary target
    df['Target'] = (df['future_return_pct'] > threshold_pct).astype(int)
    
    # Drop rows without future data
    df = df.dropna(subset=['Target'])
    
    # Distribution
    target_dist = df['Target'].value_counts()
    st.info(f"📊 Target Distribution: **UP (1):** {target_dist.get(1, 0)} | **DOWN/FLAT (0):** {target_dist.get(0, 0)} | **Ratio:** {target_dist.get(1, 0) / len(df) * 100:.1f}%")
    
    return df


# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================
@st.cache_data
def load_data(symbol, timeframe, days_back):
    """Load data from Bybit using BybitDataManager"""
    try:
        manager = BybitDataManager(symbol=symbol, interval=timeframe)
        df = manager.get_data(days_back=days_back)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


# ============================================================================
# SIDEBAR - DATA LOADING CONTROLS
# ============================================================================
st.sidebar.header("📊 Data Configuration")

symbol = st.sidebar.text_input("Symbol", value="BTCUSDT")
timeframe = st.sidebar.text_input("Timeframe (minutes)", value="15")
days_back = st.sidebar.number_input("Days Back", min_value=1, max_value=365, value=90)

load_button = st.sidebar.button("🔄 Load Data", type="primary")

# Target Configuration
st.sidebar.markdown("---")
st.sidebar.header("🎯 Target Configuration")
prediction_horizon = st.sidebar.slider("Prediction Horizon (candles)", min_value=1, max_value=20, value=4)
threshold_pct = st.sidebar.slider("Take Profit Threshold (%)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Model Configuration
st.sidebar.markdown("---")
st.sidebar.header("🤖 Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100, step=50)
max_depth = st.sidebar.slider("Max Depth", min_value=5, max_value=30, value=10)

# ============================================================================
# MAIN LOGIC
# ============================================================================

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load Data
if load_button:
    with st.spinner("🔄 Loading data from Bybit..."):
        df_raw = load_data(symbol, timeframe, days_back)
        
        if df_raw is not None and len(df_raw) > 0:
            st.session_state.df_raw = df_raw
            st.session_state.data_loaded = True
            st.session_state.model_trained = False  # Reset model when new data loaded
            st.success(f"✅ Loaded **{len(df_raw)}** candles for **{symbol}** ({timeframe}m)")
        else:
            st.error("❌ Failed to load data")

# ============================================================================
# TABS - MAIN DASHBOARD
# ============================================================================
if st.session_state.data_loaded:
    
    tab1, tab2, tab3 = st.tabs(["📈 Data Inspector", "🎯 Model Performance", "⭐ Feature Importance"])
    
    # ========================================================================
    # TAB 1: DATA INSPECTOR
    # ========================================================================
    with tab1:
        st.header("📈 Data Inspector")
        
        # Feature Engineering
        with st.spinner("🏭 Running Alpha Factory..."):
            df_features = add_features(st.session_state.df_raw)
            df_with_target = create_target(df_features, prediction_horizon, threshold_pct)
            
            # Store in session state
            st.session_state.df_features = df_with_target
        
        # Display data
        st.subheader("📋 Feature DataFrame (First 100 rows)")
        st.dataframe(df_with_target.head(100), use_container_width=True)
        
        # Price Chart
        st.subheader("💹 Close Price Chart")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_with_target.index,
            y=df_with_target['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00d4ff', width=2)
        ))
        fig_price.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Feature Summary
        st.subheader("📊 Feature Summary")
        feature_cols = [c for c in df_with_target.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'Target', 'future_close', 'future_return_pct']]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(feature_cols))
        with col2:
            st.metric("Total Samples", len(df_with_target))
        with col3:
            st.metric("Date Range", f"{df_with_target.index[0].date()} to {df_with_target.index[-1].date()}")
    
    # ========================================================================
    # TAB 2: MODEL PERFORMANCE
    # ========================================================================
    with tab2:
        st.header("🎯 Model Performance")
        
        # Train Model Button
        train_button = st.button("🚀 Train Model", type="primary")
        
        if train_button:
            with st.spinner("🤖 Training Random Forest..."):
                
                # Prepare features
                df = st.session_state.df_features.copy()
                
                # Feature columns (exclude OHLCV and target-related)
                exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target', 'future_close', 'future_return_pct']
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                
                X = df[feature_cols].values
                y = df['Target'].values
                
                # Time-series split (no shuffling!)
                split_idx = int(len(df) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.feature_cols = feature_cols
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.precision = precision
                st.session_state.recall = recall
                st.session_state.accuracy = accuracy
                st.session_state.test_df = df.iloc[split_idx:].copy()
                st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
        
        # Display Results
        if st.session_state.model_trained:
            
            # Metrics
            st.subheader("📊 Model Metrics (Test Set)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Precision",
                    f"{st.session_state.precision:.2%}",
                    help="Of all predicted UPs, how many were correct?"
                )
            with col2:
                st.metric(
                    "Recall",
                    f"{st.session_state.recall:.2%}",
                    help="Of all actual UPs, how many did we catch?"
                )
            with col3:
                st.metric(
                    "Accuracy",
                    f"{st.session_state.accuracy:.2%}",
                    help="Overall correctness"
                )
            
            # Confusion Matrix
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted DOWN', 'Predicted UP'],
                y=['Actual DOWN', 'Actual UP'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
            ))
            fig_cm.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Strategy Equity Curve
            st.subheader("💰 Strategy Equity Curve")
            
            test_df = st.session_state.test_df.copy()
            test_df['prediction'] = st.session_state.y_pred
            test_df['signal'] = test_df['prediction'].apply(lambda x: 1 if x == 1 else 0)
            
            # Calculate returns
            test_df['market_return'] = test_df['close'].pct_change()
            test_df['strategy_return'] = test_df['market_return'] * test_df['signal'].shift(1)
            
            # Cumulative returns
            test_df['market_equity'] = (1 + test_df['market_return']).cumprod()
            test_df['strategy_equity'] = (1 + test_df['strategy_return'].fillna(0)).cumprod()
            
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=test_df.index,
                y=test_df['market_equity'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig_equity.add_trace(go.Scatter(
                x=test_df.index,
                y=test_df['strategy_equity'],
                mode='lines',
                name='ML Strategy',
                line=dict(color='#00d4ff', width=2)
            ))
            fig_equity.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_title="Date",
                yaxis_title="Equity",
                hovermode='x unified'
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Performance Summary
            final_market = test_df['market_equity'].iloc[-1]
            final_strategy = test_df['strategy_equity'].iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Buy & Hold Return", f"{(final_market - 1) * 100:.2f}%")
            with col2:
                st.metric("ML Strategy Return", f"{(final_strategy - 1) * 100:.2f}%")
        
        else:
            st.info("👆 Click 'Train Model' to begin training")
    
    # ========================================================================
    # TAB 3: FEATURE IMPORTANCE (THE "SIMONS" VIEW)
    # ========================================================================
    with tab3:
        st.header("⭐ Feature Importance Analysis")
        st.markdown("*The 'Renaissance Technologies' view of what drives alpha*")
        
        if st.session_state.model_trained:
            
            # Extract feature importances
            model = st.session_state.model
            feature_cols = st.session_state.feature_cols
            importances = model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Categorize features by type
            def categorize_feature(name):
                name_lower = name.lower()
                if any(x in name_lower for x in ['rsi', 'macd', 'roc', 'ao', 'stoch', 'willr', 'cci', 'mfi']):
                    return 'Momentum'
                elif any(x in name_lower for x in ['atr', 'bb_width', 'donchian', 'volatility', 'hl_ratio']):
                    return 'Volatility'
                elif any(x in name_lower for x in ['obv', 'cmf', 'volume']):
                    return 'Volume'
                elif any(x in name_lower for x in ['adx', 'sma']):
                    return 'Trend'
                elif any(x in name_lower for x in ['log_return', 'return']):
                    return 'Lagged Returns'
                elif any(x in name_lower for x in ['hour', 'day']):
                    return 'Time'
                else:
                    return 'Other'
            
            importance_df['Category'] = importance_df['Feature'].apply(categorize_feature)
            
            # Color mapping
            color_map = {
                'Momentum': '#00d4ff',
                'Volatility': '#ff6b6b',
                'Volume': '#51cf66',
                'Trend': '#ffd43b',
                'Lagged Returns': '#da77f2',
                'Time': '#ff922b',
                'Other': '#868e96'
            }
            
            importance_df['Color'] = importance_df['Category'].map(color_map)
            
            # Horizontal Bar Chart
            st.subheader("📊 Feature Importance Ranking")
            
            fig_importance = go.Figure()
            
            for category in importance_df['Category'].unique():
                cat_df = importance_df[importance_df['Category'] == category]
                fig_importance.add_trace(go.Bar(
                    y=cat_df['Feature'],
                    x=cat_df['Importance'],
                    orientation='h',
                    name=category,
                    marker=dict(color=color_map[category]),
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                ))
            
            fig_importance.update_layout(
                template='plotly_dark',
                height=max(600, len(importance_df) * 20),
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                barmode='stack',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Top 5 Features Summary
            st.subheader("🏆 Top 5 Most Important Features")
            top_5 = importance_df.tail(5).sort_values('Importance', ascending=False)
            
            for idx, row in top_5.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{row['Feature']}**")
                with col2:
                    st.markdown(f"`{row['Category']}`")
                with col3:
                    st.markdown(f"*{row['Importance']:.4f}*")
            
            # Category Breakdown
            st.subheader("📈 Importance by Category")
            category_importance = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
            
            fig_category = go.Figure(data=[
                go.Pie(
                    labels=category_importance.index,
                    values=category_importance.values,
                    marker=dict(colors=[color_map[cat] for cat in category_importance.index]),
                    hole=0.4
                )
            ])
            fig_category.update_layout(
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Insights
            st.subheader("💡 Key Insights")
            top_category = category_importance.index[0]
            top_feature = importance_df.iloc[-1]['Feature']
            
            st.markdown(f"""
            - 🥇 **Most Important Feature:** `{top_feature}` (Importance: {importance_df.iloc[-1]['Importance']:.4f})
            - 📊 **Dominant Category:** `{top_category}` (Total Importance: {category_importance.iloc[0]:.4f})
            - 🎯 **Total Features Used:** {len(feature_cols)}
            - 🔬 **Top 10 Features Account For:** {importance_df.tail(10)['Importance'].sum() / importance_df['Importance'].sum() * 100:.1f}% of total importance
            """)
            
        else:
            st.info("👆 Train a model first to see feature importance")

else:
    # Welcome Screen
    st.info("👈 **Get Started:** Configure your data parameters in the sidebar and click 'Load Data'")
    
    st.markdown("""
    ## 🚀 Welcome to the ML Alpha Research Dashboard
    
    This platform allows you to:
    
    1. **📊 Load Market Data** - Fetch historical OHLCV data from Bybit
    2. **🏭 Engineer Features** - Generate 30+ technical indicators automatically
    3. **🤖 Train ML Models** - Use Random Forest to predict price movements
    4. **⭐ Analyze Importance** - Discover which features drive alpha
    
    ### 📚 Feature Categories:
    
    - **Momentum:** RSI, MACD, ROC, Stochastic, Williams %R, CCI, MFI
    - **Volatility:** ATR, Bollinger Bands, Donchian Channels
    - **Volume:** OBV, CMF, Volume Ratios
    - **Trend:** ADX, SMA Slope
    - **Lagged Returns:** Historical price movements
    - **Time:** Hour of day, Day of week
    
    ### 🎯 Get Started:
    1. Enter your symbol (e.g., BTCUSDT)
    2. Set timeframe and lookback period
    3. Click "Load Data"
    4. Navigate through the tabs to explore!
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>ML Alpha Research Dashboard v1.0 | Built with Streamlit & scikit-learn</p>
    <p>⚠️ For research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
