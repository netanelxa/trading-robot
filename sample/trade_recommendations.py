import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

def analyze_stock(historical_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(historical_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Calculate indicators
    df['SMA20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(close=df['close']).rsi()

    # Generate recommendation
    last_row = df.iloc[-1]
    recommendation = "HOLD"
    reasons = []

    if last_row['SMA20'] > last_row['SMA50'] and last_row['MACD'] > last_row['MACD_signal']:
        recommendation = "BUY"
        reasons.append("Short-term trend is bullish (SMA20 > SMA50)")
        reasons.append("MACD indicates bullish momentum")
    elif last_row['SMA20'] < last_row['SMA50'] and last_row['MACD'] < last_row['MACD_signal']:
        recommendation = "SELL"
        reasons.append("Short-term trend is bearish (SMA20 < SMA50)")
        reasons.append("MACD indicates bearish momentum")

    if last_row['RSI'] > 70:
        recommendation = "SELL"
        reasons.append("RSI indicates overbought conditions")
    elif last_row['RSI'] < 30:
        recommendation = "BUY"
        reasons.append("RSI indicates oversold conditions")

    return recommendation, reasons

def get_trade_recommendation(symbol, historical_data):
    recommendation, reasons = analyze_stock(historical_data)
    return {
        'symbol': symbol,
        'recommendation': recommendation,
        'reasons': reasons
    }