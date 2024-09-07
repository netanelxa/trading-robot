import os
import time
import pandas as pd
from flask import Flask, render_template, request, json, jsonify,current_app, flash, redirect, url_for, Response
from redis import Redis
from fakeredis import FakeRedis
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from trade_recommendations import get_trade_recommendation, calculate_indicators
import json
import io
import numpy as np
from opentelemetry import trace
import inspect
import requests


app = Flask(__name__)
app.jinja_env.filters['tojson'] = json.dumps
serviceName = "web-ui"
tracer = trace.get_tracer(serviceName + ".tracer")

ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5001')


# Use FakeRedis for local development
if os.environ.get('FLASK_ENV') == 'development':
    redis = FakeRedis()
else:
    redis = Redis(host=os.environ.get('REDIS_HOST', 'localhost'),
                  port=int(os.environ.get('REDIS_PORT', 6379)))

ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

# Initialize Alpaca API client
api = REST(key_id=os.environ.get('APCA_API_KEY_ID'),  # Corrected env var name
           secret_key=os.environ.get('APCA_API_SECRET_KEY'),  # Corrected env var name
           base_url='https://paper-api.alpaca.markets',
           api_version='v2')



@app.route('/')
def index():
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
    return render_template('index.html')


def fetch_and_process_data(symbol, start_date, end_date):
    bars = api.get_bars(
        symbol, 
        TimeFrame.Day, 
        start_date.isoformat(), 
        end_date.isoformat(),
        adjustment='raw'
    ).df
    # Fetch SPX data for the same period
    spx_bars = api.get_bars(
        'SPY',  # Using SPY as a proxy for SPX
        TimeFrame.Day, 
        start_date.isoformat(), 
        end_date.isoformat(),
        adjustment='raw'
    ).df

    # Ensure both dataframes have the same index and remove timezone info
    bars.index = bars.index.tz_convert(None)
    spx_bars.index = spx_bars.index.tz_convert(None)
    bars = bars.reindex(spx_bars.index)
    spx_bars = spx_bars.reindex(bars.index)
    
    # Calculate indicators
    df = calculate_indicators(bars, spx_bars)   
    
    # Convert DataFrame to dictionary
    return df.to_dict('index')


def update_cache(symbol, new_data):
    cached_data = redis.get(symbol)
    if cached_data:
        cached_data = json.loads(cached_data)
        cached_data.update(new_data)
    else:
        cached_data = new_data
    
    # Convert DataFrame index (dates) to string format
    serializable_data = {}
    for date, values in cached_data.items():
        if isinstance(date, (pd.Timestamp, datetime)):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        # Ensure all numpy values are converted to native Python types
        serializable_values = {k: v.item() if isinstance(v, np.generic) else v for k, v in values.items()}
        serializable_data[date_str] = serializable_values
    
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
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(cached_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df

@app.route('/add-stock', methods=['GET', 'POST'])
def add_stock():
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
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
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
    stocks = redis.smembers('stocks')
    return render_template('stocks.html', stocks=[stock.decode('utf-8') for stock in stocks])


def get_cached_data(key, fetch_func):
    cached_data = redis.get(key)
    if cached_data:
        return json.loads(cached_data)
    else:
        data = fetch_func()
        redis.set(key, json.dumps(data))  # Set without expiration
        return data


def process_data(data_dict, days):
    end_date = datetime.now().date() - timedelta(days=1)
    cutoff_date = (end_date - timedelta(days=days))
    
    # Convert string dates to datetime objects
    data_dict = {pd.to_datetime(k).date(): v for k, v in data_dict.items()}
    
    filtered_data = {k: v for k, v in data_dict.items() if k >= cutoff_date}
    return [{
        'date': date.strftime('%Y-%m-%d'),
        'Close': item['Close'],
        'Volume': item['Volume'],
        'SPX_Close': item['SPX_Close']
    } for date, item in sorted(filtered_data.items(), reverse=True)]


@app.route('/stock/<symbol>')
def stock_detail(symbol):
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
    historical_data = get_stock_data(symbol)
    if historical_data.empty:
        return render_template('error.html', error_message="No data available for this stock.")

    data = {
        '1M': process_data(historical_data.to_dict(orient='index'), 30),
        '3M': process_data(historical_data.to_dict(orient='index'), 90),
        '1Y': process_data(historical_data.to_dict(orient='index'), 365)
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

    news_data = get_cached_data(f"{symbol}_news", fetch_news)  # Cache for 30 minutes

    # Get trade recommendation
    recommendation = get_trade_recommendation(symbol, historical_data)

    # Check for candlestick patterns on the last day
    pattern_columns = ['CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLSHOOTINGSTAR',
                       'CDLHARAMI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
                       'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLSPINNINGTOP']
    
    detected_patterns = []
    for col in pattern_columns:
        if last_data[col] != 0:
            pattern_name = col.replace('CDL', '').title()
            pattern_value = last_data[col]
            open_price = last_data['Open']
            high_price = last_data['High']
            low_price = last_data['Low']
            close_price = last_data['Close']
            
            # Pre-calculate values for the template
            price_range = high_price - low_price
            body_height = abs(close_price - open_price) / price_range * 50
            wick_top_height = (high_price - max(open_price, close_price)) / price_range * 50
            wick_bottom_height = (min(open_price, close_price) - low_price) / price_range * 50
            
            detected_patterns.append({
                'name': pattern_name,
                'value': pattern_value,
                'bullish': close_price > open_price,
                'body_height': body_height,
                'wick_top_height': wick_top_height,
                'wick_bottom_height': wick_bottom_height,
                'body_position': (high_price - max(open_price, close_price)) / price_range * 50
            })
    # ML prediction
    #'rf' or 'lr', 'svr', 'xgb', 'lstm'
    current_model_type = redis.get('current_model_type')
    if current_model_type:
        current_model_type = current_model_type.decode('utf-8')
        try:
            response = requests.post(f"{ML_SERVICE_URL}/predict", 
                                     json={"data": historical_data.to_dict(orient='index'),
                                           "model_type": current_model_type})
            if response.status_code == 200:
                prediction_data = response.json()
                prediction = prediction_data['prediction']
                confidence = prediction_data.get('confidence_interval')
                forecast = prediction_data.get('forecast')
            else:
                prediction = None
                confidence = None
                forecast = None
        except requests.RequestException:
            prediction = None
            confidence = None
            forecast = None
    else:
        prediction = None
        confidence = None
        forecast = None

    # Fetch model metrics and feature importance
    model_metrics = redis.get('model_metrics')
    if model_metrics:
        model_metrics = json.loads(model_metrics.decode('utf-8'))
    else:
        model_metrics = None

    feature_importance = redis.get('feature_importance')
    if feature_importance:
        feature_importance = json.loads(feature_importance.decode('utf-8'))
        # Sort feature importance in descending order
        feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    else:
        feature_importance = None
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
                           detected_patterns=detected_patterns,
                           recommendation=recommendation,
                           prediction=prediction,
                           confidence=confidence,
                           forecast=forecast,
                           current_model_type=current_model_type,
                           model_metrics=model_metrics,
                           feature_importance=feature_importance)


@app.route('/get_current_model')
def get_current_model():
    current_model_type = redis.get('current_model_type')
    return jsonify({"model_type": current_model_type.decode('utf-8') if current_model_type else None})


@app.route('/export/<symbol>')
def export_data(symbol):
    historical_data = get_stock_data(symbol)
    if not historical_data:
        return "No data available for this stock.", 404
    # Convert the historical data to a DataFrame
    df = pd.DataFrame.from_dict(historical_data, orient='index')
    df.index = pd.to_datetime(df.index)  # Ensure the index is datetime
    df.sort_index(ascending=False, inplace=True)  # Sort by date descending
    df.index.name = 'Date'
    
    # Reorder the columns as requested
    columns_order = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
        'SPX_Close',
        'SMA20', 'SMA50', 'SMA150', 'SMA200',
        'MACD', 'MACD_signal', 'RSI', 'CCI',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_percentage',
        'EMA20', 'HT_TRENDLINE',
        'STOCH_K', 'STOCH_D',
        'WILLR',
        'AD', 'OBV',
        'ATR', 'NATR',
        'CDLDOJI', 'CDLHAMMER', 'CDLENGULFING', 'CDLSHOOTINGSTAR',
        'CDLHARAMI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
        'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLSPINNINGTOP'
    ]
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



@app.route('/market-movers')
def market_movers():
    try:
        # Fetch top gainers and losers
        url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()

        top_gainers = data.get('top_gainers', [])[:10]  # Limit to top 10
        top_losers = data.get('top_losers', [])[:10]  # Limit to top 10
        print(f"top_gainers {top_gainers}")
        # Fetch data for some major stocks to check for 52-week highs/lows
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'JPM', 'JNJ', 'V', 'PG']
        stocks_at_high = []
        stocks_at_low = []

        for symbol in symbols:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
            response = requests.get(url)
            stock_data = response.json()

            if 'Time Series (Daily)' in stock_data:
                daily_data = stock_data['Time Series (Daily)']
                dates = list(daily_data.keys())
                current_price = float(daily_data[dates[0]]['4. close'])
                
                # Check last 252 trading days (approximately 1 year)
                high_52week = max(float(daily_data[date]['2. high']) for date in dates[:252])
                low_52week = min(float(daily_data[date]['3. low']) for date in dates[:252])

                if current_price >= high_52week * 0.99:  # Within 1% of 52-week high
                    stocks_at_high.append({'symbol': symbol, 'price': current_price})
                elif current_price <= low_52week * 1.01:  # Within 1% of 52-week low
                    stocks_at_low.append({'symbol': symbol, 'price': current_price})

        return render_template('market_movers.html', 
                            top_gainers=top_gainers, 
                            top_losers=top_losers, 
                            stocks_at_high=stocks_at_high, 
                            stocks_at_low=stocks_at_low)
    
    except requests.RequestException as e:
            # Log the error
            app.logger.error(f"Error fetching data from Alpha Vantage: {str(e)}")
            # Return an error page
            return render_template('error.html', message="Unable to fetch market data. Please try again later."), 500



def train_ml_model(model_type='rf'):
    all_stocks = redis.smembers('stocks')
    stock_data = {}
    for stock in all_stocks:
        stock_symbol = stock.decode('utf-8')
        stock_data[stock_symbol] = get_stock_data(stock_symbol).to_dict(orient='index')
    
    try:
        response = requests.post(f"{ML_SERVICE_URL}/train", 
                                 json={"data": stock_data, "model_type": model_type})
        if response.status_code == 200:
            result = response.json()
            # Store the trained model type and metrics in Redis
            redis.set('current_model_type', model_type)
            redis.set('model_metrics', json.dumps(result['metrics']))
            return {
                "message": "Model trained successfully",
                "model_type": model_type,
                "metrics": result['metrics'],
                "feature_importance": result.get('feature_importance')
            }
        else:
            return {"error": f"Error training model: {response.json().get('error')}"}
    except requests.RequestException as e:
        return {"error": f"Error connecting to ML service: {str(e)}"}
    


@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    ticker = request.json.get('ticker')
    model_type = request.json.get('model_type', 'rf')  # Default to 'rf' if not specified
    try:
        result = train_ml_model(ticker, model_type)
        return jsonify({
            "message": f"Model {model_type} trained successfully for {ticker}",
            "result": result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ml_training')
def ml_training():
    return render_template('ml_training.html')



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