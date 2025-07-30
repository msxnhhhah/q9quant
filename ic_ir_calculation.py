import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Load the CSV file
output_folder = 'real_time'
df_ic = pd.read_csv(os.path.join(output_folder, 'alphalens_data.csv'))
df_ic['Timestamp'] = pd.to_datetime(df_ic['Timestamp'])
df_ic.set_index('Timestamp', inplace=True)

# Calculate forward returns
df_ic['forward_return_1h'] = df_ic['price'].pct_change(1).shift(-1)
df_ic['forward_return_5h'] = df_ic['price'].pct_change(5).shift(-5)
df_ic['forward_return_10h'] = df_ic['price'].pct_change(10).shift(-10)

# Update periods for IC calculation
periods = [
    ('1H', df_ic['factor'], df_ic['forward_return_1h']),
    ('5H', df_ic['factor'], df_ic['forward_return_5h']),
    ('10H', df_ic['factor'], df_ic['forward_return_10h'])
]

# Function to calculate IC metrics
def calculate_ic_metrics(signal, returns, period_name):
    # Drop NaN values
    valid_data = pd.concat([signal, returns], axis=1).dropna()
    signal = valid_data.iloc[:, 0]
    returns = valid_data.iloc[:, 1]
    
    # Calculate Spearman IC (rank correlation)
    ic, p_value = stats.spearmanr(signal, returns)
    
    # Calculate metrics
    ic_mean = ic
    ic_std = np.std(signal)  # Standard deviation of signal
    risk_adjusted_ic = ic_mean / ic_std if ic_std != 0 else np.nan
    t_stat = ic * np.sqrt(len(signal) - 2) / np.sqrt(1 - ic**2)
    ic_skew = stats.skew(signal)
    ic_kurtosis = stats.kurtosis(signal)
    
    return {
        'Period': period_name,
        'IC Mean': ic_mean,
        'IC Std': ic_std,
        'Risk-Adjusted IC': risk_adjusted_ic,
        'T-Stat': t_stat,
        'P-Value': p_value,
        'IC Skew': ic_skew,
        'IC Kurtosis': ic_kurtosis
    }

# Define periods for analysis
periods = [
    ('1H (factor)', df_ic['factor'], df_ic['forward_return_1h']),
    ('1H (position)', df_ic['position'], df_ic['forward_return_1h']),
    ('5H (factor)', df_ic['factor'], df_ic['forward_return_5h']),
    ('5H (position)', df_ic['position'], df_ic['forward_return_5h']),
    ('10H (factor)', df_ic['factor'], df_ic['forward_return_10h']),
    ('10H (position)', df_ic['position'], df_ic['forward_return_10h']),
]

# Calculate IC metrics for each period
ic_results = []
for period_name, signal, returns in periods:
    metrics = calculate_ic_metrics(signal, returns, period_name)
    ic_results.append(metrics)

# Create results DataFrame
results_df = pd.DataFrame(ic_results)

# Save results to CSV
results_df.to_csv('ic_analysis_results.csv', index=False)

# Print results
print("\nInformation Coefficient Analysis Results:")
print(results_df)

# Calculate Information Ratio using pf_return as strategy and BTC futures price change as benchmark
strategy_returns_hourly = df_ic['pf_return']
benchmark_returns_hourly = df_ic['price'].pct_change(1)

# Align both series and drop NaNs
aligned = pd.concat([strategy_returns_hourly, benchmark_returns_hourly], axis=1).dropna()
strategy_returns_hourly = aligned.iloc[:, 0]
benchmark_returns_hourly = aligned.iloc[:, 1]

# Calculate hourly excess returns
hourly_excess_returns = strategy_returns_hourly - benchmark_returns_hourly

# Calculate mean and std of hourly excess returns
mean_excess = hourly_excess_returns.mean()
std_excess = hourly_excess_returns.std()

# Annualize for hourly data (24*365 hours in a year)
annualization_factor = 24 * 365
ir = mean_excess / std_excess * np.sqrt(annualization_factor) if std_excess != 0 else np.nan
print(f"\nInformation Ratio (Annualized, hourly, pf_return vs BTC futures): {ir:.4f}")

# Define rolling window for IC calculation (7 days)
rolling_window = 7 * 24  # 168 hours

# Define moving average window (30 days)
ma_window = 30 * 24  # 720 hours

# Function to compute rolling Spearman IC
def rolling_spearman(x, y, window):
    rho = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        x_window = x.iloc[i - window + 1:i + 1]
        y_window = y.iloc[i - window + 1:i + 1]
        if x_window.isnull().any() or y_window.isnull().any():
            continue
        # Check for constant input
        if x_window.nunique() == 1 or y_window.nunique() == 1:
            continue  # Leave as np.nan
        rho[i], _ = stats.spearmanr(x_window, y_window)
    return pd.Series(rho, index=x.index)

# Create a figure with subplots: 6 rows (periods) x 3 columns (plots)
fig, axes = plt.subplots(6, 3, figsize=(18, 36))  # 6 periods, 3 plots each

# Plot for each period
for i, (period_name, signal, returns) in enumerate(periods):
    # Compute rolling IC
    rolling_ic = rolling_spearman(signal, returns, rolling_window)
    
    # Compute 1-month moving average of rolling IC
    ma_ic = rolling_ic.rolling(window=ma_window).mean()
    
    # Plot rolling IC and its 1-month MA
    axes[i, 0].plot(rolling_ic, label='Rolling IC (7D)')
    axes[i, 0].plot(ma_ic, label='1M MA of Rolling IC', color='orange')
    axes[i, 0].set_title(f'{period_name} Rolling IC')
    axes[i, 0].set_xlabel('Date')
    axes[i, 0].set_ylabel('Spearman IC')
    axes[i, 0].legend()
    axes[i, 0].grid(True)
    
    # Plot histogram of rolling IC
    axes[i, 1].hist(rolling_ic.dropna(), bins=30, density=True)
    axes[i, 1].set_title(f'{period_name} IC Histogram')
    axes[i, 1].set_xlabel('IC Value')
    axes[i, 1].set_ylabel('Density')
    
    # Plot Q-Q plot of rolling IC
    stats.probplot(rolling_ic.dropna(), dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f'{period_name} IC Q-Q Plot')

# Calculate and plot 3-month rolling Information Ratio (IR)
rolling_ir_window = 90 * 24  # 3 months in hours

# Compute rolling mean and std of excess returns
rolling_mean_excess = hourly_excess_returns.rolling(window=rolling_ir_window).mean()
rolling_std_excess = hourly_excess_returns.rolling(window=rolling_ir_window).std()

# Compute rolling IR (annualized)
rolling_ir = (rolling_mean_excess / rolling_std_excess) * np.sqrt(rolling_ir_window)
rolling_ir = rolling_ir.dropna()

# Plot rolling IR
plt.figure(figsize=(14, 5))
plt.plot(rolling_ir, label='3M Rolling Information Ratio')
plt.axhline(0, color='gray', linestyle='--')
plt.title('3-Month Rolling Information Ratio (Annualized, Hourly)')
plt.xlabel('Date')
plt.ylabel('Information Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'rolling_ir_3m.png'))
plt.close()

print("3-month rolling IR plot saved to 'rolling_ir_3m.png'")

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(output_folder, 'ic_analysis_plots.png'))
plt.close()

print("Plots saved to 'ic_analysis_plots.png'")