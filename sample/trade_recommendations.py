import pandas as pd
import numpy as np
import talib
from ta.trend import SMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands



def calculate_indicators(df, spx_data):
    open_data = df['open'].values
    high_data = df['high'].values
    low_data = df['low'].values
    close_data = df['close'].values
    volume_data = df['volume'].values
    
    # Convert all data to float64 to ensure compatibility with TA-Lib
    open_data = open_data.astype(np.float64)
    high_data = high_data.astype(np.float64)
    low_data = low_data.astype(np.float64)
    close_data = close_data.astype(np.float64)
    volume_data = volume_data.astype(np.float64)

    # Existing indicators (keep these as they are)
    df['SMA20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['SMA150'] = SMAIndicator(close=df['close'], window=150).sma_indicator()
    df['SMA200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(close=df['close']).rsi()
    df['CCI'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = bollinger.bollinger_wband()
    df['BB_percentage'] = bollinger.bollinger_pband()
    
    # TA-Lib indicators
    df['EMA20'] = talib.EMA(close_data, timeperiod=20)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close_data)
    slowk, slowd = talib.STOCH(high_data, low_data, close_data, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    df['WILLR'] = talib.WILLR(high_data, low_data, close_data, timeperiod=14)
    df['AD'] = talib.AD(high_data, low_data, close_data, volume_data)
    df['OBV'] = talib.OBV(close_data, volume_data)
    df['ATR'] = talib.ATR(high_data, low_data, close_data, timeperiod=14)
    df['NATR'] = talib.NATR(high_data, low_data, close_data, timeperiod=14)
    
    # New Pattern Recognition indicators
    df['CDLDOJI'] = talib.CDLDOJI(open_data, high_data, low_data, close_data)
    df['CDLHAMMER'] = talib.CDLHAMMER(open_data, high_data, low_data, close_data)
    df['CDLENGULFING'] = talib.CDLENGULFING(open_data, high_data, low_data, close_data)
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_data, high_data, low_data, close_data)
    df['CDLHARAMI'] = talib.CDLHARAMI(open_data, high_data, low_data, close_data)
    df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_data, high_data, low_data, close_data, penetration=0)
    df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open_data, high_data, low_data, close_data, penetration=0)
    df['CDLPIERCING'] = talib.CDLPIERCING(open_data, high_data, low_data, close_data)
    df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open_data, high_data, low_data, close_data, penetration=0)
    df['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open_data, high_data, low_data, close_data)

    # Add SPX close price
    df['SPX_Close'] = spx_data['close']

    # Rename columns to match the desired format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'vwap': 'VWAP'
    })
    
    return df



def analyze_stock(historical_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(historical_data, orient='index')
    
    # Convert index to datetime and remove timezone information
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    
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

    if last_row['CCI'] > 100:
        reasons.append("CCI indicates overbought conditions")
    elif last_row['CCI'] < -100:
        reasons.append("CCI indicates oversold conditions")

    return recommendation, reasons

def get_trade_recommendation(symbol, historical_data):
    recommendation, reasons = analyze_stock(historical_data)
    return {
        'symbol': symbol,
        'recommendation': recommendation,
        'reasons': reasons
    }