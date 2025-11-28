import yfinance as yf
import argparse
import os
import pandas as pd

def fetch_yfinance(symbol='BTC-USD', period_days=3650, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    
    # Use yf.download for reliable bulk download
    df = yf.download(
        tickers=symbol,
        period=f"{period_days}d",
        interval='1d',
        auto_adjust=True,
        progress=False
    )
    
    if df.empty:
        raise ValueError("No data fetched. Check ticker, period, or your internet connection.")

    df = df[['Open','High','Low','Close','Volume']]
    df.index.name = 'Date'
    
    out_path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(out_path)
    print(f"Saved {out_path}, {len(df)} rows")
    
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTC-USD')
    parser.add_argument('--period', type=int, default=3650)
    parser.add_argument('--out', default='data')
    args = parser.parse_args()
    
    fetch_yfinance(args.symbol, args.period, args.out)
