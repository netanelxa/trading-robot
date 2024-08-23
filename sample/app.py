import os
from flask import Flask, render_template, request, redirect, url_for
from redis import Redis

app = Flask(__name__)

# Connect to Redis
redis = Redis(host=os.environ.get('REDIS_HOST', 'localhost'),
              port=int(os.environ.get('REDIS_PORT', 6379)))

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))