import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
import webbrowser
import os

# Configuration Section
config = {
    'direction': 1,          # 1 for buying high/selling low, -1 for selling high/buying low
    'long_entry_above': 1.0,  # Z-score threshold for long entry
    'short_entry_above': 1.0, # Z-score threshold for short entry
    'long_exit_above': 0.0,   # Z-score threshold for long exit
    'short_exit_above': 0.0,  # Z-score threshold for short exit
    'in_sample_ratio': 2/3,  # Adjust this value to change the in-sample to out-of-sample ratio (e.g., 0.8 for 80% in-sample)
    'position_size': 1,       # Size of each position
    'frequency': '1h',        # Data frequency (e.g., '1h' for hourly)
    'fees': 0.001,            # Transaction fees
    'initial_cash': 100000,   # Starting capital
    'min_trades': 3,          # Minimum number of trades for valid results
    'annualization_factor': 365  # For Sharpe ratio calculation
}

x_values = np.arange(100, 2500, 100)  # Window sizes for SMA and stdev
y_values = np.arange(0.25, 3.25, 0.25)  # Z-score thresholds (multiplied by config thresholds)

# Create 'full_backtest' folder if it doesn't exist
output_folder = 'real_time'
os.makedirs(output_folder, exist_ok=True)

# Load data
df_binance = pd.read_csv(os.path.join(output_folder, 'binance_futures_btc_usdt_1h_last_3_years.csv'))
df_binance['Timestamp'] = pd.to_datetime(df_binance['Timestamp'])
df_binance.set_index('Timestamp', inplace=True)
df_okx = pd.read_csv(os.path.join(output_folder, 'okx_futures_btc_usdt_1h_last_3_years.csv'))
df_okx['Timestamp'] = pd.to_datetime(df_okx['Timestamp'])
df_okx.set_index('Timestamp', inplace=True)
df_btc_spot = pd.read_csv(os.path.join(output_folder, 'binance_spot_btc_usdt_1h_last_3_years.csv'))
df_btc_spot['Timestamp'] = pd.to_datetime(df_btc_spot['Timestamp'])
df_btc_spot.set_index('Timestamp', inplace=True)

# Data splitting section
start_date = df_binance.index[0]
end_date = df_binance.index[-1]
total_duration = end_date - start_date
in_sample_duration = total_duration * config['in_sample_ratio']
split_date = start_date + in_sample_duration

# Split the data into in-sample and out-of-sample
df_binance_in = df_binance[df_binance.index < split_date]
df_binance_out = df_binance[df_binance.index >= split_date]
df_okx_in = df_okx[df_okx.index < split_date]
df_okx_out = df_okx[df_okx.index >= split_date]

# Verify the split 
print(f"Split Date: {split_date}")
print(f"In-sample size (Binance): {len(df_binance_in)}")
print(f"Out-of-sample size (Binance): {len(df_binance_out)}")

# Precompute SMA and stdev for the entire dataset
sma_dict = {x: vbt.MA.run(df_okx['Close'], window=x).ma for x in x_values}
stdev_dict = {x: vbt.MSTD.run(df_okx['Close'], window=x).mstd for x in x_values}

# Signal generation function with configurable parameters
def custom_signal_func(close, sma, stdev, y, config):
    z_scores = (close - sma) / stdev
    long_entries = np.zeros(len(close), dtype=bool)
    long_exits = np.zeros(len(close), dtype=bool)
    short_entries = np.zeros(len(close), dtype=bool)
    short_exits = np.zeros(len(close), dtype=bool)
    position = 0
    n = len(close)
    for i in range(n):
        if np.isnan(z_scores[i]):
            continue
        if i == n - 1 and position != 0:
            if position == 1:
                long_exits[i] = True
            elif position == -1:
                short_exits[i] = True
            position = 0
            continue
        if position == 0:
            # Long entry
            long_threshold = config['long_entry_above'] * y
            if (z_scores[i] > long_threshold and config['direction'] == 1) or \
               (z_scores[i] < -long_threshold and config['direction'] == -1):
                long_entries[i] = True
                position = 1
            # Short entry
            short_threshold = config['short_entry_above'] * y
            if (z_scores[i] > short_threshold and config['direction'] == -1) or \
               (z_scores[i] < -short_threshold and config['direction'] == 1):
                short_entries[i] = True
                position = -1
        elif position == 1:
            exit_threshold = config['long_exit_above']
            if (z_scores[i] < exit_threshold and config['direction'] == 1) or \
               (z_scores[i] > -exit_threshold and config['direction'] == -1):
                long_exits[i] = True
                position = 0
        elif position == -1:
            exit_threshold = config['short_exit_above']
            if (z_scores[i] > exit_threshold and config['direction'] == 1) or \
               (z_scores[i] < -exit_threshold and config['direction'] == -1):
                short_exits[i] = True
                position = 0
    return long_entries, long_exits, short_entries, short_exits

# Simulate portfolio and evaluate for both in-sample and out-of-sample
results_in = []
results_out = []
for x in x_values:
    sma = sma_dict[x]
    stdev = stdev_dict[x]
    for y in y_values:
        # Generate signals for the entire dataset
        long_entries, long_exits, short_entries, short_exits = custom_signal_func(
            df_okx['Close'].values, sma.values, stdev.values, y, config
        )
        signals = [pd.Series(sig, index=df_okx.index).shift(1, fill_value=False) 
                  for sig in [long_entries, long_exits, short_entries, short_exits]]

        # Simulate portfolio for the entire period
        pf = vbt.Portfolio.from_signals(
            close=df_binance['Open'],
            entries=signals[0],
            exits=signals[1],
            short_entries=signals[2],
            short_exits=signals[3],
            init_cash=config['initial_cash'],
            fees=config['fees'],
            freq=config['frequency'],
            size=config['position_size']
        )

        # Split results into in-sample and out-of-sample
        for period, df_binance_period, df_okx_period in [
            ('in', df_binance_in, df_okx_in),
            ('out', df_binance_out, df_okx_out)
        ]:
            pf_period = vbt.Portfolio.from_signals(
                close=df_binance_period['Open'],
                entries=signals[0].loc[df_binance_period.index],
                exits=signals[1].loc[df_binance_period.index],
                short_entries=signals[2].loc[df_binance_period.index],
                short_exits=signals[3].loc[df_binance_period.index],
                init_cash=config['initial_cash'],
                fees=config['fees'],
                freq=config['frequency'],
                size=config['position_size']
            )
            stats = pf_period.stats()
            if stats['Total Trades'] >= config['min_trades']:
                sharpe = stats['Sharpe Ratio']  * np.sqrt(365/252)
                total_return = stats['Total Return [%]']
                max_dd = stats['Max Drawdown [%]']
                trade_count = stats['Total Trades']
            else:
                sharpe = np.nan
                total_return = np.nan
                max_dd = np.nan
                trade_count = np.nan
            result = {
                'x': x,
                'y': y,
                'sharpe': sharpe,
                'return': total_return,
                'max_dd': max_dd,
                'trade_count': trade_count
            }
            if period == 'in':
                results_in.append(result)
            else:
                results_out.append(result)

# Convert results to DataFrames
results_in_df = pd.DataFrame(results_in)
results_out_df = pd.DataFrame(results_out)

# Plot heatmaps and save to 'fullbacktest' folder
metrics = [
    ('sharpe', 'Sharpe Ratio'),
    ('return', 'Total Return [%]'),
    ('max_dd', 'Max Drawdown [%]'),
    ('trade_count', 'Total Trades')
]
metric_colormaps = {
    'return': 'viridis',
    'max_dd': 'Reds',
    'trade_count': 'Blues'
}

for results_df, period in [
    (results_in_df, "In-Sample (First 2 Years)"),
    (results_out_df, "Out-of-Sample (3rd Year)")
]:
    for metric, label in metrics:
        heatmap_df = results_df.pivot(index='x', columns='y', values=metric)
        plt.figure(figsize=(14, 8))
        cmap = metric_colormaps.get(metric)
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".2f" if metric != 'trade_count' else ".0f",
            cbar_kws={'label': label},
            cmap=cmap
        )
        plt.title(f"{label} Heatmap - {period}")
        plt.xlabel("y (z-score threshold multiplier)")
        plt.ylabel("x (window size in hours)")
        plt.tight_layout()
        # Save to 'fullbacktest' folder
        safe_period = period.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').lower()
        plt.savefig(os.path.join(output_folder, f"{metric}_heatmap_{safe_period}.png"))
        plt.show()

# Find best in-sample parameters
best_row = results_in_df.loc[results_in_df['sharpe'].idxmax()]
best_x = int(best_row['x'])
best_y = best_row['y']
print(f"Best in-sample parameters: x={best_x}, y={best_y}")

# Evaluate best parameters on out-of-sample data
sma_best = sma_dict[best_x].loc[df_okx_out.index]
stdev_best = stdev_dict[best_x].loc[df_okx_out.index]
long_entries, long_exits, short_entries, short_exits = custom_signal_func(
    df_okx_out['Close'].values, sma_best.values, stdev_best.values, best_y, config
)
signals = [pd.Series(sig, index=df_okx_out.index).shift(1, fill_value=False) 
          for sig in [long_entries, long_exits, short_entries, short_exits]]

pf_best = vbt.Portfolio.from_signals(
    close=df_binance_out['Open'],
    entries=signals[0],
    exits=signals[1],
    short_entries=signals[2],
    short_exits=signals[3],
    init_cash=config['initial_cash'],
    fees=config['fees'],
    freq=config['frequency'],
    size=config['position_size']
)

# Generate QuantStats report and save to 'fullbacktest' folder
equity_curve = pf_best.value()
strategy_returns = equity_curve.pct_change().resample('1D').sum().dropna()
btc_returns = df_btc_spot['Close'].pct_change().resample('1D').sum().dropna()

qs.reports.html(
    returns=strategy_returns,
    benchmark=btc_returns,
    output=os.path.join(output_folder, 'quantstats_report.html'),
    title='Crypto Strategy QuantStats Report',
    periods_per_year=config['annualization_factor']
)
print(f"QuantStats report saved as {os.path.join(output_folder, 'quantstats_report.html')}")
webbrowser.open(os.path.join(output_folder, 'quantstats_report.html'))

# ======================================================================
# Alphalens analysis
# ======================================================================

# 1. Compute optimal z-scores using best parameters
sma_best = vbt.MA.run(df_okx['Close'], window=best_x).ma
stdev_best = vbt.MSTD.run(df_okx['Close'], window=best_x).mstd
z_scores = (df_okx['Close'] - sma_best) / stdev_best

# 3. Calculate forward returns (1-period ahead)
#    Using Binance open prices for trading
price = df_binance['Open']

df_okx.index = pd.to_datetime(df_okx.index)
df_binance.index = pd.to_datetime(df_binance.index)

# 4. Determine the date one year before the last date
last_date = df_okx.index[-1]
one_year_ago = last_date - pd.DateOffset(years=1)

# 5. Prepare Alphalens DataFrame with discrete factor for the recent 1 year
alphalens_data = pd.DataFrame({
    'factor': z_scores.loc[one_year_ago:],
    'price': price.loc[one_year_ago:]
}, index=df_okx.loc[one_year_ago:].index)

# Stateful position logic
position = []
current_pos = 0
for f in alphalens_data['factor']:
    if current_pos == 0:
        if f > best_y:
            current_pos = 1
        elif f < -best_y:
            current_pos = -1
    elif current_pos == 1:
        if f < 0:
            current_pos = 0
    elif current_pos == -1:
        if f > 0:
            current_pos = 0
    position.append(current_pos)
alphalens_data['position'] = position

# Align equity curve to alphalens_data index (last year)
equity_curve_aligned = equity_curve.loc[alphalens_data.index]
alphalens_data['equity'] = equity_curve_aligned

# Compute portfolio return as percentage change of equity curve
alphalens_data['pf_return'] = alphalens_data['equity'].pct_change().fillna(0)

# 6. Save to CSV
alphalens_file = os.path.join(output_folder, 'alphalens_data.csv')
alphalens_data.to_csv(alphalens_file)
print(f"Alphalens data saved to {alphalens_file}")
print(f"Date range: {alphalens_data.index.min()} to {alphalens_data.index.max()}")
