import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import argparse

def create_sequence_for_next_day(X_scaled, seq_len=10):
    X_seq = X_scaled[-seq_len:]
    return np.expand_dims(X_seq, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input CSV with features')
    parser.add_argument('--model_dir', default='models', help='Directory containing models')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for LSTM')
    args = parser.parse_args()

    # Load models and scaler
    xgb = joblib.load(f'{args.model_dir}/xgb.model')
    lstm = load_model(f'{args.model_dir}/lstm.h5', compile=False)
    scaler = joblib.load(f'{args.model_dir}/scaler.pkl')

    # Load dataset
    df = pd.read_csv(args.input, parse_dates=True)
    # Automatically detect index column
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    else:
        raise ValueError("CSV must have either 'Date' or 'Datetime' column")

    df = df.sort_index()
    X = df.drop(columns=['target_return'])
    y = df['target_return']

    # Scale features
    X_scaled = scaler.transform(X)

    # Last BTC price
    last_price = df['Close'].iloc[-1]

    # --------------------------
    # XGBoost next day prediction
    # --------------------------
    xgb_input = X_scaled[-1].reshape(1, -1)
    xgb_next_return = xgb.predict(xgb_input)[0]
    xgb_next_price = last_price * np.exp(xgb_next_return)

    # --------------------------
    # LSTM next day prediction
    # --------------------------
    lstm_input = create_sequence_for_next_day(X_scaled, args.seq_len)
    lstm_next_return = lstm.predict(lstm_input, verbose=0).flatten()[0]
    lstm_next_price = last_price * np.exp(lstm_next_return)

    # --------------------------
    # Save and print results
    # --------------------------
    results = pd.DataFrame({
        'Model': ['XGBoost', 'LSTM'],
        'Predicted_return': [xgb_next_return, lstm_next_return],
        'Predicted_price': [xgb_next_price, lstm_next_price]
    })

    results.to_csv(f'{args.model_dir}/next_day_prediction.csv', index=False)
    print("Next day prediction saved to", f'{args.model_dir}/next_day_prediction.csv')
    print(results)

if __name__ == "__main__":
    main()
