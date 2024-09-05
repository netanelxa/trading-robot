import os
import time
import pandas as pd
from flask import Flask, render_template, request, json, current_app, flash, redirect, url_for, Response
from redis import Redis
from fakeredis import FakeRedis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from trade_recommendations import get_trade_recommendation, calculate_indicators
import json
import io

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


def fetch_and_process_data(symbol, start_date, end_date):
    bars = api.get_bars(
        symbol, 
        TimeFrame.Day, 
        start_date.isoformat(), 
        end_date.isoformat(),
        adjustment='raw'
    ).df
    
    # Calculate indicators
    df = calculate_indicators(bars)
    
    # Convert DataFrame to dictionary
    return df.to_dict('index')



def update_cache(symbol, new_data):
    cached_data = redis.get(symbol)
    if cached_data:
        cached_data = json.loads(cached_data)
        cached_data.update(new_data)
    else:
        cached_data = new_data
    
    # Convert Timestamp index to string
    serializable_data = {k.strftime('%Y-%m-%d') if isinstance(k, pd.Timestamp) else k: v 
                         for k, v in cached_data.items()}
    
    redis.set(symbol, json.dumps(serializable_data))


def get_stock_data(symbol):
    end_date = datetime.now().date() - timedelta(days=1)
    cached_data = redis.get(symbol)
    
    if cached_data:
        cached_data = json.loads(cached_data)
        last_cached_date = max(datetime.strptime(date, '%Y-%m-%d').date() for date in cached_data.keys())
        if last_cached_date < end_date:
            start_date = last_cached_date + timedelta(days=1)
            new_data = fetch_and_process_data(symbol, start_date, end_date)
            update_cache(symbol, new_data)
            cached_data.update(new_data)
    else:
        start_date = end_date - timedelta(days=365)
        cached_data = fetch_and_process_data(symbol, start_date, end_date)
        update_cache(symbol, cached_data)
    
    return cached_data

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


def process_data(data_dict, days):
    end_date = datetime.now().date() - timedelta(days=1)
    cutoff_date = (end_date - timedelta(days=days))
    
    # Convert string dates to datetime objects
    data_dict = {pd.to_datetime(k).date(): v for k, v in data_dict.items()}
    
    filtered_data = {k: v for k, v in data_dict.items() if k >= cutoff_date}
    return [{'date': date.strftime('%Y-%m-%d'), **item} for date, item in sorted(filtered_data.items(), reverse=True)]

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    historical_data = get_stock_data(symbol)
    data = {
        '1M': process_data(historical_data, 30),
        '3M': process_data(historical_data, 90),
        '1Y': process_data(historical_data, 365)
    }

    # Get latest available price
    current_price = api.get_latest_trade(symbol).price

    # Calculate metrics
    high_52week = max(item['High'] for item in historical_data.values())
    low_52week = min(item['Low'] for item in historical_data.values())
    
    # Calculate moving averages (already in the data)
    last_data = list(historical_data.values())[-1]
    ma_50 = last_data['SMA50']
    ma_200 = last_data['SMA200']
    
    # Calculate percent change
    prev_close = list(historical_data.values())[-2]['Close']
    percent_change = ((current_price - prev_close) / prev_close) * 100

    # Fetch news (unchanged)
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
                           end_date=datetime.now().date() - timedelta(days=1),
                           high_52week=high_52week,
                           low_52week=low_52week,
                           ma_50=ma_50,
                           ma_200=ma_200,
                           percent_change=percent_change,
                           news=news_data,
                           recommendation=recommendation)


@app.route('/export/<symbol>')
def export_data(symbol):
    historical_data = get_stock_data(symbol)
    
    # Convert the historical data to a DataFrame
    df = pd.DataFrame.from_dict(historical_data, orient='index')
    df.index = pd.to_datetime(df.index)  # Ensure the index is datetime
    df.sort_index(ascending=False, inplace=True)  # Sort by date descending
    df.index.name = 'Date'
    columns_order = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'SMA20', 'SMA50', 'SMA150', 'SMA200', 'MACD', 'MACD_signal', 'RSI', 'CCI','BB_upper','BB_middle','BB_lower','BB_width','BB_percentage']
    df = df[columns_order]
    
    # Create a buffer to store the CSV data
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    
    # Create a response with the CSV data
    return Response(
        buffer.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename={symbol}_historical_data.csv'}
    )


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