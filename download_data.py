import yfinance as yf
import pandas_ta as ta
import pandas as pd
import os

def generate_market_csvs():
    print("Downloading last 60 days of 5-minute META data...")
    df = yf.download("META", period="60d", interval="5m")
    
    # Flatten the MultiIndex FIRST
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = df.columns.astype(str)
    
    print("Calculating Technical Indicators...")
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.vwap(append=True)
    
    df.dropna(inplace=True)
    
    # --- THE FIX: Bulletproof Dynamic Renaming ---
    # We dynamically search the generated columns so it doesn't matter 
    # if your version of pandas-ta adds decimals to the column names or not.
    rename_mapping = {"Close": "price", "close": "price"}
    
    for col in df.columns:
        if "EMA_" in col: rename_mapping[col] = "ema_20"
        elif "RSI_" in col: rename_mapping[col] = "rsi_14"
        elif "MACDh_" in col: rename_mapping[col] = "macd_histogram"
        elif "BBL_" in col: rename_mapping[col] = "bb_lower"
        elif "BBU_" in col: rename_mapping[col] = "bb_upper"
        elif "VWAP_" in col: rename_mapping[col] = "vwap"
        
    df = df.rename(columns=rename_mapping)
    
    # Keep only the essential features
    features = ["price", "ema_20", "rsi_14", "macd_histogram", "bb_lower", "bb_upper", "vwap"]
    df = df[features]
    
    os.makedirs("data", exist_ok=True)
    
    try:
        bull_df = df.loc["2026-03-02"] 
        sideways_df = df.loc["2026-03-11"]
        bear_df = df.loc["2026-03-26"] 
        
        bull_df.to_csv("data/bull_market.csv")
        sideways_df.to_csv("data/sideways_market.csv")
        bear_df.to_csv("data/bear_market.csv")
        
        print("\nSuccessfully saved CSVs to /data folder!")
        print(f"Rows per episode: Bull({len(bull_df)}), Sideways({len(sideways_df)}), Bear({len(bear_df)})")
        
    except KeyError as e:
        print(f"\nError: Could not find exact date in the dataset. {e}")

if __name__ == "__main__":
    generate_market_csvs()