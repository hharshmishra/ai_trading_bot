import pandas_ta as ta
import ccxt
import pandas as pd

# Connect to exchange (example: Binance)
exchange = ccxt.binance()

# Fetch OHLCV data (symbol, timeframe, limit)
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

bb = ta.bbands(df["close"], length=20, std=2)
print(bb.head())
print(bb.columns)