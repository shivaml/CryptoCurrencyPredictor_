import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
import argparse
import os

def build_features(csv_in, csv_out, n_lags=5, predict_horizon=1):

    # Load CSV and skip the first two junk rows
    df = pd.read_csv(csv_in, skiprows=2)

    # Strip spaces
    df.columns = [c.strip() for c in df.columns]

    # Rename the columns to standard OHLCV
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Convert datetime and set index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    df = dropna(df)

    # Add technical indicators
    df_ta = add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

    # Returns & target
    df_ta['return'] = df_ta['Close'].pct_change()
    df_ta['log_return'] = np.log(df_ta['Close'] / df_ta['Close'].shift(1))
    df_ta['target_return'] = df_ta['log_return'].shift(-predict_horizon)

    # Lag features
    for lag in range(1, n_lags + 1):
        df_ta[f'lag_close_{lag}'] = df_ta['Close'].shift(lag)
        df_ta[f'lag_ret_{lag}'] = df_ta['log_return'].shift(lag)

    df_ta = df_ta.dropna()

    # Select final features
    keep_cols = ['Open','High','Low','Close','Volume','log_return','target_return']
    ta_cols = [c for c in df_ta.columns if any(prefix in c for prefix in 
              ['momentum', 'trend', 'volatility', 'volume_', 'oscillator', 'vol_'])]
    lag_cols = [c for c in df_ta.columns if c.startswith('lag_')]
    feature_cols = keep_cols + ta_cols + lag_cols
    feature_cols = [c for c in feature_cols if c in df_ta.columns]
    features_df = df_ta[feature_cols].copy()

    # Save CSV
    os.makedirs(os.path.dirname(csv_out) or '.', exist_ok=True)
    features_df.to_csv(csv_out)
    print(f"✅ Saved features to {csv_out} — shape: {features_df.shape}")
    return features_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/BTC-USD-features.csv')
    parser.add_argument('--lags', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=1)
    args = parser.parse_args()
    build_features(args.input, args.output, n_lags=args.lags, predict_horizon=args.horizon)
