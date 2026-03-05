import pandas as pd
import ta
import os

data_path = "data/"
output_path = "data/processed/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

files = os.listdir(data_path)

for file in files:
    if file.endswith(".csv"):
        print(f"Processing {file}...")
        
        df = pd.read_csv(os.path.join(data_path, file))

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Handle missing values
        df.ffill(inplace=True)

        # Technical Indicators
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_14'] = df['Close'].rolling(window=14).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()

        # Target (next day close)
        df['Target'] = df['Close'].shift(-1)

        df.dropna(inplace=True)

        df.to_csv(os.path.join(output_path, file), index=False)

print("Preprocessing complete!")