import ccxt
import time
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import os

# Initialize the OKX exchange
exchange = ccxt.okx()

# Get the current time in UTC
current_time =  start_time = datetime.now(timezone.utc).replace(tzinfo=None)

# Calculate the start time (3 years ago from now)
start_time = current_time - relativedelta(years=3)

# Convert to timestamps in milliseconds
start_timestamp = int(start_time.timestamp() * 1000)
end_timestamp = int(current_time.timestamp() * 1000)

# Fetch OHLCV data
ohlcv_data = []
current_timestamp = start_timestamp

while current_timestamp < end_timestamp:
    try:
        klines = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', current_timestamp)
        if not klines:
            break
        ohlcv_data.extend(klines)
        current_timestamp = klines[-1][0] + 1  # Move to the next timestamp
        time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Convert to DataFrame
df = pd.DataFrame(ohlcv_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

# Filter by date range to ensure accuracy
df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= current_time)]
df.reset_index(drop=True, inplace=True)

# Create 'real_time' folder if it doesn't exist
if not os.path.exists('real_time'):
    os.makedirs('real_time')

# Set the file path to save under 'real_time' folder
file_path = os.path.join('real_time', 'okx_futures_btc_usdt_1h_last_3_years.csv')

# Save to CSV
df.to_csv(file_path, index=False)
print(f"Data saved to '{file_path}'")