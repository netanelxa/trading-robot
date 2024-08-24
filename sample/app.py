import os
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

def get_stock_data(symbol):
    try:
        print(f"Attempting to fetch data for {symbol}")
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=30)
        
        # Format dates as strings in YYYY-MM-DD format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Date range: {start_date_str} to {end_date_str}")

        bars = api.get_bars(symbol, TimeFrame.Day, start=start_date_str, end=end_date_str).df
        print(f"Retrieved {len(bars)} bars of data")

        # Convert to list of dictionaries and sort by date (most recent first)
        data = [
            {
                'date': index.strftime('%Y-%m-%d'),
                'close': row['close']
            }
            for index, row in bars.iterrows()
        ]
        data.sort(key=lambda x: x['date'], reverse=True)
        
        print(f"Processed and sorted {len(data)} data points")

        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error args: {e.args}")
        return None

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    current_app.logger.info(f"Entering stock_detail function for symbol: {symbol}")
    try:
        stock_data = get_stock_data(symbol)
        current_app.logger.info(f"Stock data retrieved: {stock_data}")
        if not stock_data:
            current_app.logger.warning(f"No stock data found for {symbol}")
            return render_template('error.html', message=f"Unable to fetch data for {symbol}"), 400
        return render_template('stock_detail.html', symbol=symbol, data=stock_data)
    except Exception as e:
        current_app.logger.error(f"Error in stock_detail: {str(e)}", exc_info=True)
        return render_template('error.html', message=f"An error occurred: {str(e)}"), 500


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