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
st.title("📊 加密貨幣價格波動與價值分布分析工具 (Binance API)")

# === Sidebar: 幣種選擇與時間範圍 ===
st.sidebar.header("選擇參數")

# 獲取 Binance 交易對清單
@st.cache_data(ttl=3600)  # 緩存1小時
def get_binance_symbols():
    """獲取 Binance 所有 USDT 交易對"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        if 'symbols' not in data:
            st.error("❌ 無法獲取交易對數據，請檢查 API 是否正常")
            return None
        # 過濾出 USDT 交易對且狀態為 TRADING
        usdt_symbols = []
        for symbol_info in data['symbols']:
            if ('USDT' in symbol_info['symbol']) and symbol_info['status'] == 'TRADING':
                usdt_symbols.append({
                    'symbol': symbol_info['symbol'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                })
        
        return sorted(usdt_symbols, key=lambda x: x['baseAsset'])
    except Exception as e:
        st.error(f"❌ 獲取交易對失敗: {e}")
        return None
def get_symbols_from_file():
    """從本地文件獲取交易對清單"""
    try:
        with open('coin_list.json', 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"❌ 無法讀取本地文件: {e}")
        return []
# 獲取歷史K線數據
def get_binance_klines(symbol, interval):
    """
    獲取 Binance K線數據
    interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if isinstance(data, list):
            # 轉換為 DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 數據類型轉換
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # 時間戳轉換
            df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            return df
        else:
            st.error(f"API 返回錯誤: {data}")
            return None
            
    except Exception as e:
        st.error(f"❌ 獲取數據失敗: {e}")
        return None

# 獲取當前價格
def get_current_price(symbol):
    """獲取當前價格"""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {'symbol': symbol}
        response = requests.get(url, params=params)
        data = response.json()
        return float(data['price'])
    except:
        return None

# 載入交易對
symbols_data = get_symbols_from_file()
if not symbols_data:
    st.error("❌ 無法載入交易對數據")
    st.stop()

# 創建選擇選項
symbol_options = {f"{item['baseAsset']} ({item['symbol']})": item['symbol'] 
                 for item in symbols_data}

# 預設選擇 BTC
default_symbol = "BTCUSDT"
default_key = next((k for k, v in symbol_options.items() if v == default_symbol), 
                  list(symbol_options.keys())[0])

selected_symbol_key = st.sidebar.selectbox(
    "選擇交易對", 
    list(symbol_options.keys()),
    index=list(symbol_options.keys()).index(default_key)
)
selected_symbol = symbol_options[selected_symbol_key]

# 時間範圍選擇
time_options = {
    "1天": ("1m", 1440),
    "3天": ("15m", 72), 
    "7天": ("1h", 168),
    "30天": ("4h", 180),
    "90天": ("4h", 720),
    "180天": ("1d", 180),
    "1年": ("1d", 365),
    "2年": ("1d", 730),
    "3年": ("1d", 1095),
    "5年": ("1d", 1825),
}

selected_period = st.sidebar.selectbox(
    "選擇時間範圍", 
    list(time_options.keys()),
    index=list(time_options.keys()).index("180天")
)

interval, limit = time_options[selected_period]

# === 獲取數據 ===
with st.spinner("正在獲取數據..."):
    df = get_binance_klines(selected_symbol, interval)
    current_price = get_current_price(selected_symbol)

if df is None:
    st.error("❌ 無法獲取數據，請檢查網絡連接或稍後再試")
    st.stop()

# === 顯示基本信息 ===
st.subheader(f"📉 {selected_symbol} 最新價格")
if current_price:
    price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
    color = "🟢" if price_change >= 0 else "🔴"
    st.markdown(f"- **最新價格**: ${current_price:.6f}")
    st.markdown(f"- **24h變化**: {color} {price_change:.2f}%")
else:
    st.markdown(f"- **最新價格**: ${df['close'].iloc[-1]:.6f}")
# === 價格分布圖（成交量加權 KDE） ===
st.subheader("📊 價格分布圖 (成交量加權)")

prices = df['close'].dropna()
volumes = df['volume'].loc[prices.index]

# 移除零成交量的數據點
valid_mask = volumes > 0
prices_clean = prices[valid_mask]
volumes_clean = volumes[valid_mask]

if len(prices_clean) > 10:  # 確保有足夠的數據點
    try:
        # 使用成交量作為權重的 KDE
        kde = gaussian_kde(prices_clean, weights=volumes_clean/volumes_clean.sum())
        x_vals = np.linspace(prices_clean.min(), prices_clean.max(), 1000)
        kde_vals = kde(x_vals)
        
        # 尋找峰值和谷值
        peaks = argrelextrema(kde_vals, np.greater, order=20)[0]
        troughs = argrelextrema(kde_vals, np.less, order=20)[0]
        
        fig2 = go.Figure()
        
        # KDE 曲線
        fig2.add_trace(go.Scatter(
            x=x_vals, y=kde_vals, 
            fill='tozeroy', 
            mode='lines',
            line_color='orange', 
            name='價格密度分布',
            fillcolor='rgba(255,165,0,0.3)'
        ))
        
        # 當前價格線
        current_display_price = current_price if current_price else prices.iloc[-1]
        fig2.add_vline(
            x=current_display_price, 
            line_dash="dash", 
            line_color="white", 
            line_width=3,
            annotation_text=f"當前價格: ${current_display_price:.6f}", 
            annotation_position="top right"
        )
        
        # 標記重要價格水平（峰值 - 支撐阻力位）
        for i, p in enumerate(peaks[:5]):  # 最多顯示5個峰值
            price_level = x_vals[p]
            fig2.add_vline(x=price_level, line_dash="dot", line_color="blue", line_width=1)
            fig2.add_annotation(
                x=price_level, y=kde_vals[p]*1.1, 
                text=f"阻力: ${price_level:.6f}", 
                showarrow=True, arrowhead=1,
                arrowcolor="blue", font=dict(size=10)
            )
        
        # 標記支撐位（谷值）
        for i, t in enumerate(troughs[:3]):  # 最多顯示3個谷值
            price_level = x_vals[t]
            fig2.add_vline(x=price_level, line_dash="dot", line_color="gray", line_width=1)
            fig2.add_annotation(
                x=price_level, y=kde_vals[t]*0.5, 
                text=f"支撐: ${price_level:.6f}", 
                showarrow=True, arrowhead=1,
                arrowcolor="gray", font=dict(size=10)
            )
        
        fig2.update_layout(
            height=500, 
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="價格 (USDT)", 
            yaxis_title="加權密度",
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"繪製價格分布圖時發生錯誤: {e}")
        st.write("使用簡化的價格分布圖...")
        
        fig2_simple = px.histogram(
            x=prices, 
            nbins=50, 
            title="價格分布 (簡化版)",
            labels={'x': '價格 (USDT)', 'y': '頻次'}
        )
        st.plotly_chart(fig2_simple, use_container_width=True)

# === 計算波動率統計 ===
if interval == "1d":
    # 日線數據直接計算日報酬率
    df['daily_return'] = df['close'].pct_change()
    volatility_data = df['daily_return'].dropna()
    period_name = "日"
elif interval in ["1m", "3m", "5m", "15m", "30m"]:
    # 分鐘線數據，需要重新採樣到日線
    daily_df = df['close'].resample('1D').last()
    daily_returns = daily_df.pct_change().dropna()
    volatility_data = daily_returns
    period_name = "日"
elif interval in ["1h", "4h"]:
    # 小時線數據，需要重新採樣到日線
    daily_df = df['close'].resample('1D').last()
    daily_returns = daily_df.pct_change().dropna()
    volatility_data = daily_returns
    period_name = "日"
else:
    # 其他時間框架
    volatility_data = df['close'].pct_change().dropna()
    period_name = "週期"

mean_vol = volatility_data.mean()
std_vol = volatility_data.std()
latest_vol = volatility_data.iloc[-1]

# === 互動式波動分布圖 ===
st.subheader(f"📈 {period_name}波動分布圖")

x = np.linspace(volatility_data.min(), volatility_data.max(), 500)
y = norm.pdf(x, mean_vol, std_vol)

fig1 = go.Figure()

# 添加正態分布線
fig1.add_trace(go.Scatter(
    x=x, y=y, 
    mode='lines', 
    name='正態分布', 
    line=dict(color='red', width=2)
))

# 添加統計線
fig1.add_vline(x=mean_vol, line_dash="dash", line_color="blue", 
               annotation_text="均值", annotation_position="bottom left")
fig1.add_vline(x=mean_vol + std_vol, line_dash="dash", line_color="green", 
               annotation_text="+1σ", annotation_position="bottom left")
fig1.add_vline(x=mean_vol - std_vol, line_dash="dash", line_color="green", 
               annotation_text="-1σ", annotation_position="bottom left")
fig1.add_vline(x=mean_vol + 2*std_vol, line_dash="dash", line_color="red", 
               annotation_text="+2σ", annotation_position="bottom left")
fig1.add_vline(x=mean_vol - 2*std_vol, line_dash="dash", line_color="red", 
               annotation_text="-2σ", annotation_position="bottom left")
fig1.add_vline(x=latest_vol, line_dash="dot", line_color="orange", line_width=3,
               annotation_text=f"最新: {latest_vol:.2%}", annotation_position="top right")

fig1.update_layout(
    height=500, 
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title=f"{period_name}波動率 (%)", 
    yaxis_title="密度",
    template="plotly_white",
    showlegend=True
)

st.plotly_chart(fig1, use_container_width=True)


# === 統計摘要 ===
st.subheader("📌 統計摘要")

extreme_threshold = 2 * std_vol
current_display_price = current_price if current_price else df['close'].iloc[-1]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**波動率統計**")
    st.markdown(f"- 平均{period_name}波動率: {mean_vol:.3%}")
    st.markdown(f"- 波動率標準差: {std_vol:.3%}")
    st.markdown(f"- 極端波動門檻 (±2σ): {extreme_threshold:.3%}")
    st.markdown(f"- 最新{period_name}波動率: {latest_vol:.3%}")

with col2:
    st.markdown("**價格預測區間**")
    st.markdown(f"- 當前價格: ${current_display_price:.6f}")
    st.markdown(f"- 68%信賴區間: ${current_display_price * (1 - std_vol):.6f}$ ~ ${current_display_price * (1 + std_vol):.6f}$")
    st.markdown(f"- 95%信賴區間: ${current_display_price * (1 - 2*std_vol):.6f}$ ~ ${current_display_price * (1 + 2*std_vol):.6f}$")

# === 價格趨勢圖 ===
st.subheader("📈 價格趨勢圖")

fig3 = go.Figure()

fig3.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='價格'
))

fig3.update_layout(
    height=400,
    xaxis_title="時間",
    yaxis_title="價格 (USDT)",
    template="plotly_white",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig3, use_container_width=True)

# === 成交量圖 ===
st.subheader("📊 成交量趨勢")

fig4 = go.Figure()

fig4.add_trace(go.Bar(
    x=df.index,
    y=df['volume'],
    name='成交量',
    marker_color='lightblue'
))

fig4.update_layout(
    height=300,
    xaxis_title="時間",
    yaxis_title="成交量",
    template="plotly_white"
)

st.plotly_chart(fig4, use_container_width=True)

# === 風險提示 ===
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **風險提示**")
st.sidebar.markdown("本工具僅供分析參考，不構成投資建議。加密貨幣投資存在高風險，請謹慎決策。")

# 添加數據源信息
st.sidebar.markdown("---")
st.sidebar.markdown("📡 **數據源**: Binance API")
st.sidebar.markdown(f"🕐 **更新時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")