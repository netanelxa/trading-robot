import os
import time
import pandas as pd
from flask import Flask, render_template, request,json, current_app, flash,redirect, url_for
from redis import Redis
from fakeredis import FakeRedis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from trade_recommendations import get_trade_recommendation
import json

app = Flask(__name__)
app.jinja_env.filters['tojson'] = json.dumps

# Use FakeRedis for local development
if os.environ.get('FLASK_ENV') == 'development':
    redis = FakeRedis()
else:
    redis = Redis(host=os.environ.get('REDIS_HOST', 'localhost'),
                  port=int(os.environ.get('REDIS_PORT', 6379)))


# Initialize Alpaca API client
api = REST(key_id=os.environ.get('APCA_API_KEY_ID'),  # Corrected env var name
           secret_key=os.environ.get('APCA_API_SECRET_KEY'),  # Corrected env var name
           base_url='https://paper-api.alpaca.markets',
           api_version='v2')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add-stock', methods=['GET', 'POST'])
def add_stock():
    message = None
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        if redis.sadd('stocks', symbol):
            message = f"Stock {symbol} added successfully!"
        else:
            message = f"Stock {symbol} is already being tracked."
    return render_template('add_stock.html', message=message)


@app.route('/stocks')
def list_stocks():
    stocks = redis.smembers('stocks')
    return render_template('stocks.html', stocks=[stock.decode('utf-8') for stock in stocks])


def fetch_historical_data(symbol, start_date, end_date):
    def fetch_func():
        # Fetch historical bars
        bars = api.get_bars(
            symbol, 
            TimeFrame.Day, 
            start_date.isoformat(), 
            end_date.isoformat(),
            adjustment='raw'
        ).df

        # Convert the DataFrame to a dictionary, converting Timestamp index to string
        historical_data = {date.strftime('%Y-%m-%d'): {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'trade_count': row['trade_count'],
            'vwap': row['vwap']
        } for date, row in bars.iterrows()}

        return historical_data

    cache_key = f"{symbol}_historical_{start_date.isoformat()}_{end_date.isoformat()}"
    return get_cached_data(cache_key, fetch_func, expiry=24*3600)  # Cache for 24 hours

def get_cached_data(key, fetch_func, expiry=3600):
    cached_data = redis.get(key)
    if cached_data:
        return json.loads(cached_data)
    else:
        data = fetch_func()
        redis.setex(key, expiry, json.dumps(data))
        return data



@app.route('/stock/<symbol>')
def stock_detail(symbol):
    end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
    start_date_1y = end_date - timedelta(days=365)
    historical_data = fetch_historical_data(symbol, start_date_1y, end_date)
    # Process the data for different time periods
    def process_data(data_dict, days):
        cutoff_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
        filtered_data = {k: v for k, v in data_dict.items() if k >= cutoff_date}
        sorted_data = sorted(filtered_data.items(), key=lambda x: x[0], reverse=True)
        return [{
            'date': date,
            'open': item['open'],
            'high': item['high'],
            'low': item['low'],
            'close': item['close'],
            'volume': item['volume'],
            'vwap': item['vwap']
        } for date, item in sorted_data]

    data = {
        '1M': process_data(historical_data, 30),
        '3M': process_data(historical_data, 90),
        '1Y': process_data(historical_data, 365)
    }
    # Get latest available price with caching
    def fetch_latest_price():
        latest_trade = api.get_latest_trade(symbol)
        return latest_trade.price

    current_price = get_cached_data(f"{symbol}_latest_price", fetch_latest_price, expiry=300)  # Cache for 5 minutes

    # Calculate metrics
    high_52week = max(item['high'] for item in historical_data.values())
    low_52week = min(item['low'] for item in historical_data.values())
    
    # Calculate moving averages
    closing_prices = [item['close'] for item in reversed(data['1Y'])]  # Reverse to get chronological order
    ma_50 = sum(closing_prices[-50:]) / 50 if len(closing_prices) >= 50 else None
    ma_200 = sum(closing_prices[-200:]) / 200 if len(closing_prices) >= 200 else None
    
    # Calculate percent change
    prev_close = data['1Y'][1]['close']  # Second item is the previous day's close
    percent_change = ((current_price - prev_close) / prev_close) * 100


    # Fetch news with caching
    def fetch_news():
        news = api.get_news(symbol, limit=5)
        return [{
            'headline': item.headline,
            'summary': item.summary,
            'article_url': item.url,
            'published_at': item.created_at.isoformat()
        } for item in news]

    news_data = get_cached_data(f"{symbol}_news", fetch_news, expiry=1800)  # Cache for 30 minutes

    # Get trade recommendation
    recommendation = get_trade_recommendation(symbol, historical_data)
    return render_template('stock_detail.html', 
                            symbol=symbol, 
                            data=data,
                            current_price=current_price,
                            end_date=end_date,
                            high_52week=high_52week,
                            low_52week=low_52week,
                            ma_50=ma_50,
                            ma_200=ma_200,
                            percent_change=percent_change,
                            news=news_data,
                            recommendation=recommendation)



# Debug function to check API connection
def check_api_connection():
    try:
        account = api.get_account()
        print(f"Successfully connected to Alpaca API. Account status: {account.status}")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {str(e)}")

# Call this function when your app starts
check_api_connection()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)),debug=True)