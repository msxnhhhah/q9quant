import ccxt
import time
import pandas as pd

exchange = ccxt.binance()

# Set the start and end dates
start_date = '2021-01-01T00:00:00Z'
end_date = '2023-12-31T23:59:59Z'

# Convert dates to timestamps
start_timestamp = exchange.parse8601(start_date)
end_timestamp = exchange.parse8601(end_date)

# Fetch data
ohlcv_data = []
current_timestamp = start_timestamp

while current_timestamp < end_timestamp:
    try:
        klines = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', current_timestamp)
        if not klines:
            break
        ohlcv_data.extend(klines)
        current_timestamp = klines[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Convert to DataFrame and filter by date range
df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
start_dt = pd.to_datetime(start_date).to_datetime64()
end_dt = pd.to_datetime(end_date).to_datetime64()
df = df[(df['Timestamp'] >= start_dt) & (df['Timestamp'] <= end_dt)]
df.reset_index(drop=True, inplace=True)
df.to_csv('binance_futures_btc_usdt_1h_2021_2023.csv', index=False)
