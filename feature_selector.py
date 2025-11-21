"""
Sniper Feature Selection Pipeline
=================================
Selects the Top 25 features for a "Sniper" strategy (High Precision, Short Duration).

Logic:
1. Labeling: Triple Barrier Method (+2% TP, -1% SL, 36 bars limit).
2. Filtering: Variance Threshold & Correlation Filter.
3. Ranking: Mutual Information.
4. Validation: Random Forest Classifier.

Author: Senior Quant Data Scientist
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def get_triple_barrier_labels(df, tp=0.02, sl=0.01, horizon=36):
    """
    Generates Triple Barrier labels.
    Class 1: Price hits TP before SL and before Horizon.
    Class 0: Price hits SL, or Time Limit reached.
    """
    print(f"  [LABELING] Generating Triple Barrier Labels (TP={tp}, SL={sl}, Horizon={horizon})...")
    labels = []
    
    # Convert to numpy for speed
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    n = len(df)
    
    for i in range(n):
        # Cannot label the last 'horizon' bars correctly
        if i + horizon >= n:
            labels.append(np.nan)
            continue
            
        entry_price = close[i]
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        
        outcome = 0 # Default to Fail (Time Limit or SL)
        
        # Look forward
        for j in range(1, horizon + 1):
            current_high = high[i + j]
            current_low = low[i + j]
            
            # Check TP first (Optimistic? No, usually check High/Low logic)
            # Strict logic: If Low hits SL, it's a loss. If High hits TP, it's a win.
            # What if both happen in same candle?
            # Conservative: Assume SL hit first if Low <= SL.
            
            if current_low <= sl_price:
                outcome = 0 # SL Hit
                break
            
            if current_high >= tp_price:
                outcome = 1 # TP Hit
                break
                
        labels.append(outcome)
        
    return np.array(labels)

def filter_correlated_features(X, y, threshold=0.95):
    """
    Removes highly correlated features, keeping the one with higher correlation to target.
    """
    print(f"  [FILTER] Removing features with correlation > {threshold}...")
    
    # Calculate correlation with target
    correlations_with_target = X.corrwith(pd.Series(y, index=X.index)).abs()
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    
    for column in upper.columns:
        if any(upper[column] > threshold):
            # Find features that are highly correlated with this one
            correlated_cols = upper.index[upper[column] > threshold].tolist()
            
            # Compare with current column
            for other_col in correlated_cols:
                # Keep the one with higher correlation to target
                if correlations_with_target[column] < correlations_with_target[other_col]:
                    to_drop.add(column)
                else:
                    to_drop.add(other_col)
                    
    print(f"    Dropping {len(to_drop)} redundant features.")
    return X.drop(columns=to_drop)

def main():
    print("="*70)
    print("SNIPER FEATURE SELECTION PIPELINE")
    print("="*70)
    
    # 1. Load Data
    try:
        df = pd.read_parquet('btc_1000_features.parquet')
        print(f"[INFO] Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("[ERROR] 'btc_1000_features.parquet' not found. Run signal_factory.py first.")
        return

    # 2. Labeling
    # Using parameters for "Sniper": +2% TP, -1% SL, 36 bars (3 hours on 5m? No, user said 36 bars. 
    # If 15m data, 36 bars = 9 hours. If 5m data, 36 bars = 3 hours. 
    # User context implies 15m data in dashboard, but user mentioned "3 hours" in prompt.
    # I'll stick to 36 bars as requested.)
    labels = get_triple_barrier_labels(df, tp=0.02, sl=0.01, horizon=36)
    
    df['target'] = labels
    
    # Drop NaNs (last horizon rows)
    df_labeled = df.dropna(subset=['target']).copy()
    
    # Check Class Balance
    n_class_1 = (df_labeled['target'] == 1).sum()
    n_class_0 = (df_labeled['target'] == 0).sum()
    print(f"  [LABELS] Class 1 (Sniper Hit): {n_class_1} | Class 0 (Fail): {n_class_0}")
    print(f"  [LABELS] Win Rate: {n_class_1 / len(df_labeled):.2%}")
    
    if n_class_1 < 10:
        print("[WARNING] Not enough positive samples to train. Adjust TP/SL or get more data.")
        # Proceeding anyway but results will be poor
        
    # 3. Prepare X and y
    # Exclude OHLCV and target from features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'timestamp', 'target']
    feature_cols = [c for c in df_labeled.columns if c not in exclude_cols]
    
    X = df_labeled[feature_cols]
    y = df_labeled['target']
    
    print(f"\n[INFO] Starting Feature Selection on {len(feature_cols)} features...")
    
    # 4. Variance Threshold (Remove constants)
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    X_var = X.loc[:, selector.get_support()]
    print(f"  [FILTER] Variance Threshold dropped {len(feature_cols) - X_var.shape[1]} features.")
    
    # 5. Correlation Filter
    X_corr = filter_correlated_features(X_var, y, threshold=0.95)
    
    # 6. Mutual Information Ranking
    print("\n  [RANKING] Computing Mutual Information (this may take a moment)...")
    mi_scores = mutual_info_classif(X_corr, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_corr.columns)
    mi_series = mi_series.sort_values(ascending=False)
    
    top_25_features = mi_series.head(25).index.tolist()
    print(f"  [RESULT] Top 25 Features Selected.")
    print(top_25_features)
    
    # 7. Validation
    print("\n[INFO] Validating with Random Forest...")
    X_final = X_corr[top_25_features]
    
    # Time-series split (no shuffle)
    split_idx = int(len(X_final) * 0.8)
    X_train, X_test = X_final.iloc[:split_idx], X_final.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  [METRICS] Precision: {prec:.4f} | Recall: {rec:.4f} | Accuracy: {acc:.4f}")
    
    if prec < 0.5:
        print("  [WARNING] Precision is low. The strategy might be too aggressive or needs more data.")
        
    # 8. Visualization
    print("\n[INFO] Generating Feature Importance Plot...")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': top_25_features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True) # For horizontal bar chart
    
    fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                 title='Top 25 Sniper Features (RF Importance)',
                 color='Importance', color_continuous_scale='Viridis')
    
    # Save plot (optional, or just show if in notebook, but here we are in script)
    # We will save it as HTML
    fig.write_html("sniper_feature_importance.html")
    print("  [Saved] sniper_feature_importance.html")
    
    # 9. Save Final Dataset
    final_cols = ['open', 'high', 'low', 'close', 'volume'] + top_25_features + ['target']
    df_final = df_labeled[final_cols]
    df_final.to_parquet('btc_sniper_ready.parquet')
    print(f"\n[SUCCESS] Saved {len(df_final)} rows to 'btc_sniper_ready.parquet'")
    print("="*70)

if __name__ == "__main__":
    main()
