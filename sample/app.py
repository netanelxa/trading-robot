import os
import time
from flask import Flask, render_template, request,current_app, flash,redirect, url_for
from redis import Redis
from fakeredis import FakeRedis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
import json

app = Flask(__name__)

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
    start_date_1m = end_date - timedelta(days=30)
    start_date_3m = end_date - timedelta(days=90)
    start_date_1y = end_date - timedelta(days=365)

    # Fetch historical data with caching
    def fetch_historical_data(start_date):
        def fetch_func():
            data = api.get_bars(
                symbol, 
                TimeFrame.Day, 
                start_date.isoformat(), 
                end_date.isoformat(),
                adjustment='raw'
            ).df
            # Convert DataFrame to a dictionary with date strings as keys
            return {date.strftime('%Y-%m-%d'): row.to_dict() for date, row in data.iterrows()}
        
        cache_key = f"{symbol}_historical_{start_date.isoformat()}_{end_date.isoformat()}"
        return get_cached_data(cache_key, fetch_func, expiry=24*3600)  # Cache for 24 hours

    data_1m = fetch_historical_data(start_date_1m)
    data_3m = fetch_historical_data(start_date_3m)
    data_1y = fetch_historical_data(start_date_1y)

    def process_data(data_dict):
        sorted_data = sorted(data_dict.items(), key=lambda x: x[0], reverse=True)
        return [{'date': date, 'close': item['close']} for date, item in sorted_data]


    data = {
        '1M': process_data(data_1m),
        '3M': process_data(data_3m),
        '1Y': process_data(data_1y)
    }

    # Get latest available price with caching
    def fetch_latest_price():
        latest_trade = api.get_latest_trade(symbol)
        return latest_trade.price

    current_price = get_cached_data(f"{symbol}_latest_price", fetch_latest_price, expiry=300)  # Cache for 5 minutes

    # Calculate metrics
    high_52week = max(item['high'] for item in data_1y.values())
    low_52week = min(item['low'] for item in data_1y.values())
    
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
                           news=news_data)


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