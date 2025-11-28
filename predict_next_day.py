import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def create_sequence_for_next_day(X_scaled, seq_len=10):
    """
    Returns last seq_len rows as a 3D array for LSTM prediction.
    Input: X_scaled -> numpy array of shape (n_samples, n_features)
    Output: array of shape (1, seq_len, n_features)
    """
    X_seq = X_scaled[-seq_len:]
    return np.expand_dims(X_seq, axis=0)

def main():
    # Load models and scaler
    xgb = joblib.load('models/xgb.model')
    lstm = load_model('models/lstm.h5', compile=False)
    scaler = joblib.load('models/scaler.pkl')

    # Load full dataset with features
    df = pd.read_csv('data/BTC-USD_realtime_20251002_195839.csv', index_col='Date', parse_dates=True)
    df = df.sort_index()  # Ensure chronological order
    X = df.drop(columns=['target_return'])
    y = df['target_return']

    # Scale features
    X_scaled = scaler.transform(X)

    # Use actual last BTC close price for calculation
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
    seq_len = 10
    lstm_input = create_sequence_for_next_day(X_scaled, seq_len)
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

    results.to_csv('models/next_day_prediction.csv', index=False)
    print("Next day prediction saved to models/next_day_prediction.csv")
    print(results)

if __name__ == "__main__":
    main()
