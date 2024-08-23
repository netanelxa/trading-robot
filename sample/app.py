import os
from flask import Flask, render_template, request, redirect, url_for
from redis import Redis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta


app = Flask(__name__)

# Connect to Redis
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



def get_stock_data(symbol, days=30):
    """
    Fetch stock data for the given symbol for the last 'days' number of days.
    Returns a list of dictionaries containing date and closing price.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        bars = api.get_bars(symbol, TimeFrame.Day, start_date, end_date).df
        return [{"date": index.strftime('%Y-%m-%d'), "close": row['close']} 
                for index, row in bars.iterrows()]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return []

# Example usage in a Flask route
@app.route('/stock/<symbol>')
def stock_detail(symbol):
    stock_data = get_stock_data(symbol)
    if not stock_data:
        # Handle the error, maybe return an error page or message
        return render_template('error.html', message="Unable to fetch stock data"), 400
    return render_template('stock_detail.html', symbol=symbol, data=stock_data)

# Debug function to check API connection
def check_api_connection():
    print(f"APCA_API_KEY_ID: {os.environ.get('APCA_API_KEY_ID')}")
    print(f"APCA_API_SECRET_KEY: {os.environ.get('APCA_API_SECRET_KEY')}")
    try:
        account = api.get_account()
        print(f"Successfully connected to Alpaca API. Account status: {account.status}")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {str(e)}")

# Call this function when your app starts
check_api_connection()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))