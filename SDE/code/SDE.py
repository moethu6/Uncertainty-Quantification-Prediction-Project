import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === CONFIG === #
DATA_DIR = 'Processed_Datasets_600'  # folder containing one CSV per stock
TRADING_DAYS = 252
GDPR_PHASES = {
    'Pre-GDPR': ('2015-04-01', '2015-12-31'),
    'During-GDPR': ('2016-01-01', '2018-05-24'),
    'Post-GDPR': ('2018-05-25', '2029-12-31')
}

def estimate_gbm_params(price_series):
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean() * TRADING_DAYS
    sigma = log_returns.std() * np.sqrt(TRADING_DAYS)
    return mu, sigma

# === MAIN PIPELINE === #
results = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        filepath = os.path.join(DATA_DIR, filename)
        print(filepath)
        df = pd.read_csv(filepath, parse_dates=['datadate'])
        df.set_index('datadate', inplace=True)
        stock_name = filename.replace('.csv', '')

        for phase, (start, end) in GDPR_PHASES.items():
            try:
                prices = df.loc[start:end]['prccd']
                if len(prices) < 100:
                    print(f"Not enough data for {start, end}")  # skip if not enough data

                mu, sigma = estimate_gbm_params(prices)
                results.append({
                    'Stock': stock_name,
                    'Period': phase,
                    'Mu (Drift)': mu,
                    'Sigma (Volatility)': sigma
                })
            except:
                continue  # skip bad data

# === OUTPUT === #
results_df = pd.DataFrame(results)
pivoted = results_df.pivot(index='Stock', columns='Period', values=['Mu (Drift)', 'Sigma (Volatility)'])
pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
pivoted.reset_index(inplace=True)

# Save
pivoted.to_csv('gdpr_sde_output.csv', index=False)

# Optional: visualize changes in volatility
plt.figure()
pivoted[['Sigma (Volatility)_Pre-GDPR', 'Sigma (Volatility)_During-GDPR', 'Sigma (Volatility)_Post-GDPR']].boxplot()
plt.title('Volatility Changes Around GDPR')
plt.ylabel('Annualized Volatility (σ)')
plt.show()
