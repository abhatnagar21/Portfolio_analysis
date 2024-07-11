import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(stocks, start='2020-01-01', end='2023-01-01')['Adj Close']
returns = data.pct_change().dropna()
weights = np.array([0.25,0.25, 0.25, 0.25])
preturn = np.sum(weights * returns.mean()) * 252
#print(preturn)
pvolatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))#computation of variance
rfrate = 0.01 
sharperatio = (preturn - rfrate) / pvolatility
#print(sharperatio)
clevel = 0.95
vr = np.percentile(returns.dot(weights), (1 - clevel) * 100)
#print(vr)
marketdata = yf.download('^GSPC', start='2020-01-01', end='2023-01-01')['Adj Close']
#print(marketdata)
marketreturns = marketdata.pct_change().dropna()
#print(marketreturns)
beta = np.cov(returns.dot(weights), marketreturns)[0, 1] / np.var(marketreturns)
#print(beta)
print(f'Annual Return: {preturn:.2%}')
print(f'Annual Volatility: {pvolatility:.2%}')
print(f'Sharpe Ratio: {sharperatio:.2f}')
print(f'Value at Risk (VaR) at 95% confidence level: {vr:.2%}')
print(f'Portfolio Beta: {beta:.2f}')
plt.figure(figsize=(10, 6))
plt.plot((1 + returns.dot(weights)).cumprod(), label='Portfolio')
plt.plot((1 + marketreturns).cumprod(), label='Market (S&P 500)')
plt.title('Portfolio vs Market Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
#plotting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Portfolio Stocks')
plt.show()
plt.figure(figsize=(10, 6))
plt.bar(returns.columns, weights)
plt.title('Portfolio Weights')
plt.xlabel('Stock')
plt.ylabel('Weight')
plt.show()
