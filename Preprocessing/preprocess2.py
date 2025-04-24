import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load CSV for stock data and XLSX for ETF data

company = "Vodafone"
stock_df = pd.read_csv(f'Datasets/{company}.csv', parse_dates=['datadate'], index_col='datadate')
etf_df = pd.read_excel('Datasets/stoxx50.xlsx', parse_dates=['datadate'], index_col='datadate')

# Remove duplicates if any exist
stock_df = stock_df[~stock_df.index.duplicated(keep='first')]
etf_df = etf_df[~etf_df.index.duplicated(keep='first')]

# Merge data on Date (inner join)
df = pd.DataFrame({
    'Stock': stock_df['prccd'],
    'ETF': etf_df['prccd']
}).dropna()

# Calculate daily log returns
df['Stock_Return'] = np.log(df['Stock'] / df['Stock'].shift(1))
df['ETF_Return'] = np.log(df['ETF'] / df['ETF'].shift(1))
df.dropna(inplace=True)

# Linear regression to estimate market impact
X = df['ETF_Return'].values.reshape(-1, 1)
y = df['Stock_Return'].values
reg = LinearRegression().fit(X, y)
beta = reg.coef_[0]
alpha = reg.intercept_

# Compute residuals (idiosyncratic returns)
df['Residual'] = y - (alpha + beta * df['ETF_Return'].values)

# Reconstruct denoised price
df['Denoised_Price'] = np.exp(df['Residual'].cumsum()) * df['Stock'].iloc[0]

# Replace 'prccd' with denoised prices in the original DataFrame
stock_df['prccd'] = df['Denoised_Price']

# Save the updated stock DataFrame to a new CSV
stock_df.to_csv(f'denoised_{company}.csv')

# Calculate the daily percentage change between denoised and original stock prices
df['Price_Change'] = ((df['Denoised_Price'] - df['Stock']) / df['Stock']) * 100

# Calculate the average percentage change
average_change = df['Price_Change'].mean()


print(f"Average percentage change between denoised and original stock prices: {average_change:.4f}%")
print("New CSV file 'denoised_stock_data.csv' generated with denoised prices.")
print(alpha, beta)
