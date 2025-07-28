import json
import requests
def get_coin_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    return requests.get(url).json()
def get_binance_symbols():
    """獲取 Binance 所有 USDT 交易對"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        # 過濾出 USDT 交易對且狀態為 TRADING
        usdt_symbols = []
        for symbol_info in data['symbols']:
            if ('USDT' in symbol_info['symbol']) and symbol_info['status'] == 'TRADING':
                usdt_symbols.append({
                    'symbol': symbol_info['symbol'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                })
                print(f"Found USDT trading pair: {symbol_info['symbol']} with base asset {symbol_info['baseAsset']}")
        
        return sorted(usdt_symbols, key=lambda x: x['baseAsset'])
    except Exception as e:
        raise e

coin_list = get_binance_symbols()
print(f"Total USDT trading pairs: {len(coin_list)}")
with open('coin_list.json', 'w') as f:
    json.dump(coin_list, f)