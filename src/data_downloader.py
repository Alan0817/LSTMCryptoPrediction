import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


# Download BTC historical data
df = yf.download(
    "BTC-USD",
    start="2020-01-01",
    end="2025-01-01",
    interval="1d"
)

print(df.head())
print(df.shape)

# Flatten multi-level columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Remove missing values
df = df.dropna()

# Keep only useful columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print(df.isnull().sum())

print(type(df['Close']))
print(df['Close'].shape)
print(df.columns)

# Daily return
df['Return'] = df['Close'].pct_change()

# Future return (target reference)
df['Future_Return'] = df['Close'].pct_change().shift(-1)


# Predict whether next-day return is positive
threshold = 0.005
df['Target'] = (df['Future_Return'] > threshold).astype(int)


# Add technical indicators
# RSI → momentum / overbought signal
# volatility → market uncertainty
# MACD → trend momentum
# moving averages → trend smoothing

# RSI
df['RSI'] = RSIIndicator(close=df['Close']).rsi()

# MACD
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

# Bollinger Bands
bb = BollingerBands(close=df['Close'])
df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()

# Add Moving Averages
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

df['EMA_7'] = df['Close'].ewm(span=7).mean()

# Add Valatility 
df['Volatility'] = df['Return'].rolling(window=7).std()

# Final clean up
df = df.dropna()

print(df.head())
print(df.columns)
print(df.shape)

# Save the data
df.to_csv("data/btc_processed.csv")

print("Processed dataset saved.")