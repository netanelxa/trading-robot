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
import threading
import uuid
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


app.jinja_env.filters['tojson'] = json.dumps
serviceName = "web-ui"
tracer = trace.get_tracer(serviceName + ".tracer")

ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://ml-service:5002')

print(f"ML Service URL: {ML_SERVICE_URL}")  # Add this line for debugging


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


def fetch_and_process_data(symbol, start_date=None, end_date=None):
    end_date = end_date or (datetime.now().date() - timedelta(days=1))  # Use yesterday's date
    start_date = start_date or (end_date - timedelta(days=365))

    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    try:
        bars = api.get_bars(
            symbol, 
            TimeFrame.Day, 
            start=start_date.isoformat(), 
            end=end_date.isoformat(),
            adjustment='raw'
        ).df
        
        print(f"Bars shape: {bars.shape}")
        print(f"Bars head:\n{bars.head()}")
        
        # Fetch SPX data for the same period
        spx_bars = api.get_bars(
            'SPY',  # Using SPY as a proxy for SPX
            TimeFrame.Day, 
            start=start_date.isoformat(), 
            end=end_date.isoformat(),
            adjustment='raw'
        ).df
        
        print(f"SPX bars shape: {spx_bars.shape}")
        print(f"SPX bars head:\n{spx_bars.head()}")
        
        if bars.empty or spx_bars.empty:
            print("Error: One or both DataFrames are empty")
            return None

        # Ensure both dataframes have the same index and remove timezone info
        bars.index = bars.index.tz_convert(None)
        spx_bars.index = spx_bars.index.tz_convert(None)
        bars = bars.reindex(spx_bars.index)
        spx_bars = spx_bars.reindex(bars.index)
        
        # Calculate indicators
        df = calculate_indicators(bars, spx_bars)   
        
        # Convert DataFrame to dictionary
        return df.to_dict('index')
    except api.rest.APIError as e:
        print(f"Alpaca API Error: {str(e)}")
        return None

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
    end_date = datetime.now().date() - timedelta(days=1)  # Use yesterday's date
    cached_data = redis.get(symbol)
    
    if cached_data:
        cached_data = json.loads(cached_data)
        df = pd.DataFrame.from_dict(cached_data, orient='index')
        df.index = pd.to_datetime(df.index)
        last_cached_date = df.index.max().date()
        if last_cached_date < end_date:
            start_date = last_cached_date + timedelta(days=1)
            new_data = fetch_and_process_data(symbol, start_date, end_date)
            if new_data:
                new_df = pd.DataFrame.from_dict(new_data, orient='index')
                df = pd.concat([df, new_df])
                update_cache(symbol, df.to_dict(orient='index'))
    else:
        start_date = end_date - timedelta(days=365)
        data = fetch_and_process_data(symbol, start_date, end_date)
        if data:
            df = pd.DataFrame.from_dict(data, orient='index')
            update_cache(symbol, df.to_dict(orient='index'))
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data was fetched
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    return df

def fetch_stock_data_background(symbol):
    print(f"Fetching data for {symbol} in the background")
    get_stock_data(symbol)
    print(f"Finished fetching data for {symbol}")

@app.route('/add-stock', methods=['GET', 'POST'])
def add_stock():
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
    message = None
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        if redis.sadd('stocks', symbol):
            message = f"Stock {symbol} added successfully!"
            threading.Thread(target=fetch_stock_data_background, args=(symbol,)).start()
        else:
            message = f"Stock {symbol} is already being tracked."
    return render_template('add_stock.html', message=message)

@app.route('/stocks')
def list_stocks():
    stocks = [stock.decode('utf-8') for stock in redis.smembers('stocks')]
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(stocks)
    return render_template('stocks.html', stocks=stocks)

def get_cached_data(key, fetch_func):
    cached_data = redis.get(key)
    if cached_data:
        return json.loads(cached_data)
    else:
        data = fetch_func()
        redis.set(key, json.dumps(data))  # Set without expiration
        return data


def process_data(df):
    return [{
        'date': date.strftime('%Y-%m-%d'),
        'Close': row['Close'],
        'Volume': int(row['Volume']),
    } for date, row in df.iterrows()]

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    historical_data = get_stock_data(symbol)
    if historical_data.empty:
        return render_template('error.html', error_message="No data available for this stock.")
    historical_data = historical_data.sort_index(ascending=False)

    data = {
        '1M': process_data(historical_data.head(30)),
        '3M': process_data(historical_data.head(90)),
        '1Y': process_data(historical_data)
    }

    # Get latest available price
    current_price = api.get_latest_trade(symbol).price

    # Calculate metrics
    high_52week = historical_data['High'].max()
    low_52week = historical_data['Low'].min()
    
    # Calculate moving averages (already in the data)
    last_data = historical_data.iloc[0]  # Most recent data point
    ma_50 = last_data['SMA50']
    ma_200 = last_data['SMA200']
    
    # Calculate percent change
    prev_close = historical_data['Close'].iloc[-2]
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

    news_data = get_cached_data(f"{symbol}_news", fetch_news)

    # Get trade recommendation
    recommendation = get_trade_recommendation(symbol, historical_data.sort_index(ascending=True))

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
    symbol_model_type = redis.get(f"{symbol}_model_type")
    if symbol_model_type:
        current_model_type = symbol_model_type.decode('utf-8')
        try:
            # Convert historical_data to a serializable format
            serializable_data = {
                date.isoformat(): {
                    key: json_serialize(value) 
                    for key, value in row.items()
                }
                for date, row in historical_data.iterrows()
            }
            
            response = requests.post(
                f"{ML_SERVICE_URL}/predict", 
                json={
                    "symbol": symbol,
                    "data": serializable_data
                },
                timeout=10  # Add a timeout to prevent hanging
            )
            print(f"Prediction response status code: {response.status_code}")
            print(f"Prediction response content: {response.text[:1000]}...")  # Print first 1000 chars of response
            
            if response.status_code == 200:
                prediction_data = response.json()
                prediction = prediction_data['prediction']
                confidence = prediction_data.get('confidence_interval')
                forecast = prediction_data.get('forecast')
                print(f"Prediction: {prediction}, Confidence: {confidence}, Forecast: {forecast}")
            else:
                print(f"Error in prediction response: {response.text}")
                prediction = None
                confidence = None
                forecast = None
        except requests.RequestException as e:
            print(f"Request exception during prediction: {str(e)}")
            prediction = None
            confidence = None
            forecast = None
    else:
        print(f"No model type found for symbol {symbol}")
        current_model_type = None
        prediction = None
        confidence = None
        forecast = None
        # Fetch model metrics and feature importance
    model_metrics_data = redis.get(f"{symbol}_model_metrics")
    if model_metrics_data:
        model_metrics = json.loads(model_metrics_data.decode('utf-8'))
        print(f"Model metrics: {model_metrics}")
    else:
        print(f"No model metrics found for symbol {symbol}")
        model_metrics = None
    feature_importance_data = redis.get(f"{symbol}_feature_importance")
    if feature_importance_data:
        feature_importance = json.loads(feature_importance_data.decode('utf-8'))
        # Sort feature importance in descending order
        feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"Feature importance (top 5): {feature_importance[:5]}")
    else:
        print(f"No feature importance found for symbol {symbol}")
        feature_importance = None
    return render_template('stock_detail.html', 
                           symbol=symbol, 
                           data=data,
                           current_price=current_price,
                           end_date=historical_data.index[0].date(),
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
    if historical_data.empty:
        return "No data available for this stock.", 404
    
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
    
    # Filter and reorder columns
    available_columns = [col for col in columns_order if col in historical_data.columns]
    df = historical_data[available_columns]
    
    df = df.sort_index(ascending=False)  # Sort by date descending
    df.index.name = 'Date'
    
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

def train_ml_model(ticker=None, model_type='rf'):
    if ticker:
        stock_data = {ticker: get_stock_data(ticker).to_dict(orient='index')}
    else:
        all_stocks = redis.smembers('stocks')
        stock_data = {}
        for stock in all_stocks:
            stock_symbol = stock.decode('utf-8')
            stock_data[stock_symbol] = get_stock_data(stock_symbol).to_dict(orient='index')
    
    # Convert Timestamp index to string and handle non-JSON-compliant floats
    for symbol, data in stock_data.items():
        stock_data[symbol] = {k.isoformat(): {kk: json_serialize(vv) for kk, vv in v.items()} for k, v in data.items()}
    
    try:
        response = requests.post(f"{ML_SERVICE_URL}/train", 
                                 json={"data": stock_data, "model_type": model_type},
                                 timeout=30)  # Add a timeout
        
        if response.status_code == 200:
            result = response.json()
            # Store the trained model type and metrics in Redis
            redis.set('current_model_type', model_type)
            redis.set('model_metrics', json.dumps(result['metrics'], default=json_serialize))
            if 'feature_importance' in result:
                redis.set('feature_importance', json.dumps(result['feature_importance'], default=json_serialize))
            return {
                "message": "Model trained successfully",
                "model_type": model_type,
                "metrics": result['metrics'],
                "feature_importance": result.get('feature_importance')
            }
        else:
            return {"error": f"Error training model: {response.text}"}
    except requests.RequestException as e:
        return {"error": f"Error connecting to ML service: {str(e)}"}


# Global dictionary to store training status
training_tasks = {}

def serialize_for_json(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



def train_model_task(task_id, symbol, model_type):
    try:
        print(f"Starting training task for symbol: {symbol}, model type: {model_type}")
        response = requests.post(f"{ML_SERVICE_URL}/train", 
                                 json={"symbol": symbol, "model_type": model_type})
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            training_tasks[task_id] = {
                "status": "completed",
                "result": result
            }
        else:
            training_tasks[task_id] = {
                "status": "failed",
                "error": f"ML service returned status code {response.status_code}: {response.text}"
            }
    except Exception as e:
        print(f"Exception in train_model_task: {str(e)}")
        training_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

def json_serialize(obj):
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return str(obj)  # Convert numeric types to string
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()  # Convert datetime objects to ISO format string
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    symbol = request.json.get('symbol')
    model_type = request.json.get('model_type', 'rf')
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    task_id = str(uuid.uuid4())  # Generate a unique task ID
    training_tasks[task_id] = {"status": "in_progress"}
    
    # Start the training in a background thread
    thread = threading.Thread(target=train_model_task, args=(task_id, symbol, model_type))
    thread.start()
    
    return jsonify({"task_id": task_id, "message": "Training started"}), 202

@app.route('/training_status/<task_id>')
def training_status(task_id):
    try:
        task = training_tasks.get(task_id)
        if not task:
            return jsonify({"status": "not_found", "message": "Task ID not found"}), 404
        
        # Use custom JSON serialization
        serialized_task = json.loads(json.dumps(task, default=json_serialize))
        return jsonify(serialized_task)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
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

@app.route('/test_ml_service')
def test_ml_service():
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            return jsonify({"message": "ML service is accessible"}), 200
        else:
            return jsonify({"error": f"ML service returned unexpected status: {response.status_code}"}), 500
    except requests.RequestException as e:
        return jsonify({"error": f"Error connecting to ML service: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)),debug=True)