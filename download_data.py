import yfinance as yf
import pandas as pd
import os

stocks = [
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL",
    "META", "NVDA", "BRK-B", "JNJ", "JPM"
]

cryptos = ["BTC-USD", "ETH-USD"]

start_date = "2018-01-01"
end_date = "2024-12-31"

if not os.path.exists("data"):
    os.makedirs("data")

def download_assets(asset_list):
    for asset in asset_list:
        print(f"Downloading {asset}...")
        
        data = yf.download(asset, start=start_date, end=end_date)

        # 🔥 สำคัญ: reset index เพื่อให้ Date เป็น column ปกติ
        data.reset_index(inplace=True)

        # ลบ multi-level columns ถ้ามี
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.to_csv(f"data/{asset}.csv", index=False)

download_assets(stocks)
download_assets(cryptos)

print("Download complete!")