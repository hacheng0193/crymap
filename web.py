import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, gaussian_kde
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import json
import time
st.set_page_config(layout="wide")
st.title("📊 加密貨幣價格波動與價值分布分析工具")

# === Sidebar: 幣種選擇與時間範圍 ===
st.sidebar.header("選擇參數")

# 取得 CoinGecko 上所有支援幣種清單
def get_coin_list():
    with open('coin_list.json', 'r') as file:
        return json.load(file)

coin_list = get_coin_list()
try:
    coin_name_map = {f"{coin['id']}": coin['id'] for coin in coin_list}
except Exception as e:
    st.error(f"❌ 取得幣種清單時發生錯誤: {e}")
    st.stop()
default_coin = coin_list[0]['id']
selected_coin = st.sidebar.selectbox("選擇幣種", sorted(coin_name_map.keys()), index=sorted(coin_name_map.keys()).index(default_coin))

options = [7, 14, 30, 90, 180, 360]
days = st.sidebar.selectbox("選擇觀察天數", options, index=options.index(180))

coin_id = coin_name_map[selected_coin]


# === Step 1: 抓取歷史資料（每15分鐘） ===
url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
params = {
    'vs_currency': 'usd',
    'days': str(days),
    #'interval': '15m'
}
resp = requests.get(url, params=params)
data = resp.json()
time.sleep(2.5)
if 'prices' not in data:
    st.error("❌ 無法取得資料，請確認幣種是否支援。")
    st.stop()

# 整理 DataFrame
df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)
df.drop(columns=['timestamp'], inplace=True)
df['volume'] = [v[1] for v in data['total_volumes']]

st.subheader(f"📉 {coin_id} 最新價格")
latest_price = df['price'].iloc[-1]
st.markdown(f"- **最新價格**：${latest_price:.5f}$")

# === Step 2: 計算報酬率與統計 ===
df_resampled = df.resample('1D').agg({'price': 'last'})
df_resampled['volatility'] = df_resampled['price'].pct_change()
volatilitys = df_resampled['volatility'].dropna()
mean = volatilitys.mean()
std = volatilitys.std()
today_volatility = volatilitys.iloc[-1]

# === Step 3: 互動式波動分布圖 ===

x = np.linspace(volatilitys.min(), volatilitys.max(), 500)
y = norm.pdf(x, mean, std)

fig1 = go.Figure()
#fig1.add_trace(go.Histogram(x=volatilitys, histnorm='probability density', nbinsx=50,marker_color='gold', opacity=0.6, name='Daily volatilitys'))
fig1.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Dist.', line=dict(color='red')))
fig1.add_vline(x=mean, line_dash="dash", line_color="blue", annotation_text="Mean", annotation_position="bottom left")
fig1.add_vline(x=mean + std, line_dash="dash", line_color="green", annotation_text="+1σ", annotation_position="bottom left")
fig1.add_vline(x=mean - std, line_dash="dash", line_color="green", annotation_text="-1σ", annotation_position="bottom left")
fig1.add_vline(x=mean + 2*std, line_dash="dash", line_color="red", annotation_text="+2σ", annotation_position="bottom left")
fig1.add_vline(x=mean - 2*std, line_dash="dash", line_color="red", annotation_text="-2σ", annotation_position="bottom left")
fig1.add_vline(x=today_volatility, line_dash="dot", line_color="white", annotation_text=f"  Today : {today_volatility:.2%}")
fig1.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                  xaxis_title="Daily Volatility(%)", yaxis_title="Density",
                  template="plotly_white")


# === Step 4: 價格分布圖（成交量加權 KDE，互動式） ===

prices = df['price'].dropna()
volumes = df['volume'].loc[prices.index]
kde = gaussian_kde(prices, weights=volumes)
x_vals = np.linspace(prices.min(), prices.max(), 1000)
kde_vals = kde(x_vals)
peaks = argrelextrema(kde_vals, np.greater)[0]
troughs = argrelextrema(kde_vals, np.less)[0]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_vals, y=kde_vals, fill='tozeroy', mode='lines',
                          line_color='orange', name='KDE Weighted'))
fig2.add_vline(x=prices.iloc[-1], line_dash="dash", line_color="white", annotation_text=f"  Today Price: {prices.iloc[-1]:.5f}", annotation_position="bottom right")

for p in peaks:
    fig2.add_vline(x=x_vals[p], line_dash="dot", line_color="blue")
    fig2.add_annotation(x=x_vals[p], y=kde_vals[p], text=f"核心: {x_vals[p]:.5f}", showarrow=True, arrowhead=1)
for t in troughs:
    fig2.add_vline(x=x_vals[t], line_dash="dot", line_color="gray")
    fig2.add_annotation(x=x_vals[t], y=kde_vals[t], text=f"錨點: {x_vals[t]:.5f}", showarrow=True, arrowhead=1)

fig2.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20),
                  xaxis_title="Price (USD)", yaxis_title="Weighted Density",
                  template="plotly_white")


# === Step 5: 統計摘要 ===
st.subheader("📌 統計摘要")
extreme_threshold = 2 * std
curr_price = df['price'].iloc[-1]
st.markdown(f"- **極端波動門檻 (±2σ)**：{extreme_threshold:.3%}")
st.markdown(f"- **今日波動率**：{today_volatility:.3%}")
st.markdown(f"- **預測區間 (68%)**：${curr_price * (1 - std):.5f}$ ~ ${curr_price * (1 + std):.5f}$")
st.markdown(f"- **預測區間 (95%)**：${curr_price * (1 - 2 * std):.5f}$ ~ ${curr_price * (1 + 2 * std):.5f}$")


st.subheader("📊 價格出現次數分布圖 (成交量加權 KDE)")
st.plotly_chart(fig2, use_container_width=True)
st.subheader("📈 日內波動分布圖")
st.plotly_chart(fig1, use_container_width=True)