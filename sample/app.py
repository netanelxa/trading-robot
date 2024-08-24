import os
import time
from flask import Flask, render_template, request,current_app, flash,redirect, url_for
from redis import Redis
from fakeredis import FakeRedis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta


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



@app.route('/stock/<symbol>')
def stock_detail(symbol):
    end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
    start_date_1m = end_date - timedelta(days=30)
    start_date_3m = end_date - timedelta(days=90)
    start_date_1y = end_date - timedelta(days=365)

    # Fetch historical data
    def get_data(start_date):
        data = api.get_bars(
            symbol, 
            TimeFrame.Day, 
            start_date.isoformat(), 
            end_date.isoformat(),
            adjustment='raw'
        ).df
        time.sleep(0.2)  # Sleep for 200ms to avoid hitting rate limits
        return data

    data_1m = get_data(start_date_1m)
    data_3m = get_data(start_date_3m)
    data_1y = get_data(start_date_1y)

    def process_data(df):
        return [{'date': date.strftime('%Y-%m-%d'), 'close': close} 
                for date, close in zip(df.index, df['close'])]

    data = {
        '1M': process_data(data_1m),
        '3M': process_data(data_3m),
        '1Y': process_data(data_1y)
    }

    # Get latest available price and calculate metrics
    latest_trade = api.get_latest_trade(symbol)
    current_price = latest_trade.price

    # Calculate metrics
    high_52week = data_1y['high'].max()
    low_52week = data_1y['low'].min()
    ma_50 = data_3m['close'].tail(50).mean()
    ma_200 = data_1y['close'].tail(200).mean()
    
    # Calculate percent change
    prev_close = data_1y.iloc[-2]['close']
    percent_change = ((current_price - prev_close) / prev_close) * 100

    # Fetch news
    news = api.get_news(symbol, limit=5)
    news_data = [{
        'headline': item.headline,
        'summary': item.summary,
        'article_url': item.url,
        'published_at': item.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for item in news]

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