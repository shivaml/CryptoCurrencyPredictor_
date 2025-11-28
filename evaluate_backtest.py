import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, callback
from tensorflow.keras import layers, models, callbacks as tf_callbacks, losses, metrics

# ---------------------------
# XGBoost Training
# ---------------------------
def train_xgboost(X_train, y_train, X_val, y_val, save_path):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        tree_method='hist',
        verbosity=0
    )
    # Early stopping callback
    es = [callback.EarlyStopping(rounds=20, save_best=True)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=es,
        verbose=False
    )
    joblib.dump(model, save_path)
    print(f"Saved XGBoost to {save_path}")
    return model

# ---------------------------
# LSTM Model
# ---------------------------
def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    # Use proper Keras loss and metric classes
    model.compile(
        optimizer='adam',
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError()]
    )
    return model

# ---------------------------
# Sequence Builder
# ---------------------------
def create_sequences(X, y, seq_len=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)].values)
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

# ---------------------------
# Main Training
# ---------------------------
def main(input_csv, model_dir):
    np.random.seed(42)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(input_csv, index_col='Datetime', parse_dates=True)
    y = df['target_return']
    X = df.drop(columns=['target_return'])

    # Split train / val / test
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # Train XGBoost
    xgb_path = os.path.join(model_dir, 'xgb.model')
    xgb = train_xgboost(X_train_s, y_train.values, X_val_s, y_val.values, xgb_path)

    # Train LSTM
    seq_len = 10
    Xs_train, ys_train = create_sequences(pd.DataFrame(X_train_s, columns=X.columns), y_train.values, seq_len)
    Xs_val, ys_val = create_sequences(pd.DataFrame(X_val_s, columns=X.columns), y_val.values, seq_len)

    lstm = build_lstm_model(Xs_train.shape[1:])
    cb = tf_callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lstm.fit(Xs_train, ys_train, validation_data=(Xs_val, ys_val),
             epochs=80, batch_size=32, callbacks=[cb], verbose=2)
    lstm.save(os.path.join(model_dir, 'lstm.h5'))
    print(f"Saved LSTM to {os.path.join(model_dir, 'lstm.h5')}")

    # Save test sets for evaluation
    pd.DataFrame(X_test_s, index=X_test.index, columns=X.columns).to_csv(os.path.join(model_dir, 'X_test_scaled.csv'))
    y_test.to_csv(os.path.join(model_dir, 'y_test.csv'))
    print("Training complete.")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/BTC-USD-features.csv')
    parser.add_argument('--model_dir', default='models')
    args = parser.parse_args()
    main(args.input, args.model_dir)
