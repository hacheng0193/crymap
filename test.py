import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
# === Step 1: 抓取歷史資料 ===
end = datetime.now()
start = end - timedelta(days=90)

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '90',
}
resp = requests.get(url, params=params)
data = resp.json()
print(data)
# 整理價格資料
df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
df.drop(columns=['timestamp'], inplace=True)

# 加入成交量資料
df['volume'] = [v[1] for v in data['total_volumes']]

# === Step 2: 計算日內漲跌幅、標準差等 ===
df['return'] = df['price'].pct_change()
returns = df['return'].dropna()
mean = returns.mean()
std = returns.std()

# === Step 3: 畫日內波動分布圖 ===
plt.figure(figsize=(10, 5))
x = np.linspace(returns.min(), returns.max(), 100)
plt.hist(returns, bins=50, density=True, alpha=0.6, color='gold')
plt.plot(x, norm.pdf(x, mean, std), 'r-', label='Normal Dist.')
plt.axvline(mean, color='blue', linestyle='--', label='Mean')
plt.axvline(mean + std, color='green', linestyle='--', label='+1σ')
plt.axvline(mean - std, color='green', linestyle='--')
plt.axvline(mean + 2 * std, color='red', linestyle='--', label='+2σ')
plt.axvline(mean - 2 * std, color='red', linestyle='--')
plt.title('BTC 日內波動分布圖')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 4: 價格分佈圖（出現次數） ===
plt.figure(figsize=(10, 5))
prices = df['price'].dropna()
kde = gaussian_kde(prices)
x_vals = np.linspace(prices.min(), prices.max(), 500)
kde_vals = kde(x_vals)

plt.fill_between(x_vals, kde_vals, color='orange', alpha=0.6)
plt.axvline(prices.iloc[-1], color='black', linestyle='--', label='Current Price')
plt.title('BTC 價格出現次數分布圖 (KDE)')
plt.xlabel('Price (USD)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 5: 輸出極端波動門檻與預測範圍 ===
extreme_threshold = 2 * std
print("極端波動門檻 (±2σ): {:.2%}".format(extreme_threshold))
print("預測範圍 (68%信賴區): {:.2f} - {:.2f}".format(
    df['price'].iloc[-1] * (1 - std), df['price'].iloc[-1] * (1 + std)))
print("預測範圍 (95%信賴區): {:.2f} - {:.2f}".format(
    df['price'].iloc[-1] * (1 - 2 * std), df['price'].iloc[-1] * (1 + 2 * std)))
