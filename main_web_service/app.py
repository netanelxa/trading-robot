import os
import time
import pandas as pd
from flask import Flask, render_template, request, json, jsonify,session,current_app, flash, redirect, url_for, Response
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
from bs4 import BeautifulSoup
import yfinance as yf
from functools import wraps
import base64
import ssl

app = Flask(__name__)
app.secret_key = os.environ.get('APP_KEY')
app.logger.setLevel(logging.INFO)


app.jinja_env.filters['tojson'] = json.dumps
serviceName = "web-ui"
tracer = trace.get_tracer(serviceName + ".tracer")

ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://ml-service:5002')
DAYS_TO_FETCH=2000
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

# Telegram Bot configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}'
TELEGRAM_CHAT_ID1 = os.environ.get('TELEGRAM_CHAT_ID1')  # You'll need to get this for each user
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD') 

@app.route('/')
def index():
    tracer.start_as_current_span(inspect.currentframe().f_code.co_name)
    return render_template('index.html')


def decode_secret(secret):
    return base64.b64decode(secret).decode('utf-8')


def fetch_and_process_data(symbol, start_date=None, end_date=None):
    end_date = end_date or (datetime.now().date() - timedelta(days=1))  # Use yesterday's date
    start_date = start_date or (end_date - timedelta(days=DAYS_TO_FETCH))

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
            # Fetch all data again to ensure proper calculations
            start_date = end_date - timedelta(days=DAYS_TO_FETCH)
            data = fetch_and_process_data(symbol, start_date, end_date)
            if data:
                df = pd.DataFrame.from_dict(data, orient='index')

                update_cache(symbol, df.to_dict(orient='index'))
    else:
        start_date = end_date - timedelta(days=DAYS_TO_FETCH)
        data = fetch_and_process_data(symbol, start_date, end_date)
        if data:
            df = pd.DataFrame.from_dict(data, orient='index')
            update_cache(symbol, df.to_dict(orient='index'))
        else:
            return pd.DataFrame()

    
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
        'SPX_Close': row['SPX_Close']
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

    (current_model_type, prediction, confidence, forecast, 
     forecast_confidence, model_metrics, feature_importance) = get_ml_prediction(symbol, historical_data)

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
                            forecast_confidence=forecast_confidence,
                            current_model_type=current_model_type,
                            model_metrics=model_metrics,
                            feature_importance=feature_importance,
                            zip=zip)


def get_ml_prediction(symbol, historical_data):
    symbol_model_type = redis.get(f"{symbol}_model_type")
    if not symbol_model_type:
        app.logger.warning(f"No model type found for symbol {symbol}")
        return None, None, None, None, None, None, None

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
            json={"symbol": symbol, "data": serializable_data},
            timeout=10
        )
        response.raise_for_status()
        
        prediction_data = response.json()
        app.logger.info(f"Prediction data received for {symbol}: {prediction_data}")
        
        return (
            current_model_type,
            prediction_data.get('prediction'),
            prediction_data.get('confidence'),
            prediction_data.get('forecast', []),
            prediction_data.get('forecast_confidence', []),
            get_model_metrics(symbol),
            get_feature_importance(symbol)
        )
    except requests.RequestException as e:
        app.logger.error(f"Request exception during prediction for {symbol}: {str(e)}")
    except json.JSONDecodeError:
        app.logger.error(f"Failed to parse prediction response for {symbol}: {response.text}")
    except Exception as e:
        app.logger.error(f"Unexpected error during prediction for {symbol}: {str(e)}")
    
    return current_model_type, None, None, [], [], None, None

def get_model_metrics(symbol):
    model_metrics_data = redis.get(f"{symbol}_model_metrics")
    if model_metrics_data:
        try:
            return json.loads(model_metrics_data.decode('utf-8'))
        except json.JSONDecodeError:
            app.logger.error(f"Failed to parse model metrics for {symbol}")
    return None

def get_feature_importance(symbol):
    feature_importance_data = redis.get(f"{symbol}_feature_importance")
    if feature_importance_data:
        try:
            feature_importance = json.loads(feature_importance_data.decode('utf-8'))
            return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        except json.JSONDecodeError:
            app.logger.error(f"Failed to parse feature importance for {symbol}")
    return None

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
        # Scrape top gainers
        gainers_url = "https://finance.yahoo.com/gainers?offset=0&count=100"
        gainers_response = requests.get(gainers_url)
        gainers_soup = BeautifulSoup(gainers_response.text, 'html.parser')
        gainers_table = gainers_soup.find('table', {'class': 'W(100%)'})
        
        print(f"Gainers table found: {gainers_table is not None}")
        
        top_gainers = []

        if gainers_table:
            rows = gainers_table.find_all('tr')
            app.logger.info(f"Number of rows in gainers table: {len(rows)}")
            
            for i, row in enumerate(rows[1:11], start=1):  # Skip header, get top 10
                cols = row.find_all('td')
                app.logger.info(f"Row {i}: Number of columns: {len(cols)}")
                
                if len(cols) >= 5:
                    try:
                        gainer = {
                            'Symbol': cols[0].text,
                            'Name': cols[1].text,
                            'Price': float(cols[2].text.replace(',', '')),
                            'Change': float(cols[3].text.replace('+', '')),
                            '% Change': float(cols[4].text.replace('+', '').replace('%', ''))
                        }
                        top_gainers.append(gainer)
                        app.logger.info(f"Added gainer: {gainer}")
                    except ValueError as ve:
                        app.logger.error(f"Error parsing row {i}: {ve}")
                else:
                    app.logger.warning(f"Row {i} has insufficient columns: {len(cols)}")
        else:
            app.logger.error("Gainers table not found")

        # Scrape top losers (similar changes as above)
        losers_url = "https://finance.yahoo.com/losers?offset=0&count=100"
        losers_response = requests.get(losers_url)
        losers_soup = BeautifulSoup(losers_response.text, 'html.parser')
        losers_table = losers_soup.find('table', {'class': 'W(100%)'})
        
        app.logger.info(f"Losers table found: {losers_table is not None}")
        
        top_losers = []

        if losers_table:
            rows = losers_table.find_all('tr')
            app.logger.info(f"Number of rows in losers table: {len(rows)}")
            
            for i, row in enumerate(rows[1:11], start=1):  # Skip header, get top 10
                cols = row.find_all('td')
                app.logger.info(f"Row {i}: Number of columns: {len(cols)}")
                
                if len(cols) >= 5:
                    try:
                        loser = {
                            'Symbol': cols[0].text,
                            'Name': cols[1].text,
                            'Price': float(cols[2].text.replace(',', '')),
                            'Change': float(cols[3].text.replace('-', '')),
                            '% Change': float(cols[4].text.replace('-', '').replace('%', ''))
                        }
                        top_losers.append(loser)
                        app.logger.info(f"Added loser: {loser}")
                    except ValueError as ve:
                        app.logger.error(f"Error parsing row {i}: {ve}")
                else:
                    app.logger.warning(f"Row {i} has insufficient columns: {len(cols)}")
        else:
            app.logger.error("Losers table not found")

        # Fetch data for some major stocks to check for 52-week highs/lows
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'JNJ', 'V', 'PG']
        stocks_at_high = []
        stocks_at_low = []

        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                current_price = info['currentPrice']
                fiftyTwoWeekHigh = info['fiftyTwoWeekHigh']
                fiftyTwoWeekLow = info['fiftyTwoWeekLow']

                if current_price >= fiftyTwoWeekHigh * 0.99:  # Within 1% of 52-week high
                    stocks_at_high.append({'symbol': symbol, 'price': current_price})
                    app.logger.info(f"Added {symbol} to stocks at high")
                elif current_price <= fiftyTwoWeekLow * 1.01:  # Within 1% of 52-week low
                    stocks_at_low.append({'symbol': symbol, 'price': current_price})
                    app.logger.info(f"Added {symbol} to stocks at low")
            except Exception as e:
                app.logger.error(f"Error processing {symbol}: {str(e)}")

        app.logger.info(f"Number of top gainers: {len(top_gainers)}")
        app.logger.info(f"Number of top losers: {len(top_losers)}")
        app.logger.info(f"Number of stocks at 52-week high: {len(stocks_at_high)}")
        app.logger.info(f"Number of stocks at 52-week low: {len(stocks_at_low)}")

        return render_template('market_movers.html', 
                               top_gainers=top_gainers, 
                               top_losers=top_losers, 
                               stocks_at_high=stocks_at_high, 
                               stocks_at_low=stocks_at_low)
    
    except Exception as e:
        # Log the error
        app.logger.error(f"Error fetching market data: {str(e)}")
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


@app.route('/download_prepared_data', methods=['GET'])
def download_prepared_data():
    try:
        # Proxy the request to the ml_service on port 5002
        response = requests.get(f"{ML_SERVICE_URL}/download_prepared_data")
        
        # Forward the response from the ml_service
        return response.content, response.status_code, response.headers.items()
    except Exception as e:
        return jsonify({"error": f"Error in proxying request: {str(e)}"}), 500

@app.route('/view_prepared_data', methods=['GET'])
def view_prepared_data():
    try:
        # Proxy the request to the ml_service on port 5002
        response = requests.get(f"{ML_SERVICE_URL}/view_prepared_data")
        
        # Forward the response from the ml_service
        return response.content, response.status_code, response.headers.items()
    except Exception as e:
        return jsonify({"error": f"Error in proxying request: {str(e)}"}), 500


# Debug function to check API connection
def check_api_connection():
    try:
        account = api.get_account()
        print(f"Successfully connected to Alpaca API. Account status: {account.status}")
    except Exception as e:
        print(f"Error connecting to Alpaca API: {str(e)}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/account')
@login_required
def account():
    account = api.get_account()
    positions = api.list_positions()
    return render_template('account.html', account=account, positions=positions)

@app.route('/buy', methods=['POST'])
@login_required
def buy_stock():
    symbol = request.form['symbol']
    qty = request.form['quantity']
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
        message = f"Buy order placed for {qty} shares of {symbol}"
        send_telegram_message(TELEGRAM_CHAT_ID1, message)
        app.logger.info(f"Buy order placed: {message}")
        return jsonify({"message": message}), 200
    except Exception as e:
        error_message = f"Error placing buy order: {str(e)}"
        send_telegram_message(TELEGRAM_CHAT_ID1, error_message)
        app.logger.error(f"Buy order error: {error_message}")
        return jsonify({"error": error_message}), 400

@app.route('/sell', methods=['POST'])
@login_required
def sell_stock():
    symbol = request.form['symbol']
    qty = request.form['quantity']
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        message = f"Sell order placed for {qty} shares of {symbol}"
        send_telegram_message(TELEGRAM_CHAT_ID1, message)
        app.logger.info(f"Sell order placed: {message}")
        return jsonify({"message": message}), 200
    except Exception as e:
        error_message = f"Error placing sell order: {str(e)}"
        send_telegram_message(TELEGRAM_CHAT_ID1, error_message)
        app.logger.error(f"Sell order error: {error_message}")
        return jsonify({"error": error_message}), 400

@app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def handle_telegram_update():
    update = request.json
    app.logger.info(f"Received Telegram update: {update}")

    if 'message' in update:
        chat_id = update['message']['chat']['id']
        text = update['message'].get('text', '')

        app.logger.info(f"Received message: '{text}' from chat_id: {chat_id}")

        if text.lower() == '/start':
            send_telegram_message(chat_id, "Welcome to your Stock Trading Bot! Use /account to see your account details.")
        elif text.lower() == '/account':
            try:
                account = api.get_account()
                message = f"Account Balance: ${account.cash}\nPortfolio Value: ${account.portfolio_value}"
                send_telegram_message(chat_id, message)
            except Exception as e:
                app.logger.error(f"Error fetching account details: {str(e)}")
                send_telegram_message(chat_id, "Sorry, there was an error fetching your account details.")
        else:
            send_telegram_message(chat_id, "Sorry, I don't understand that command. Try /start or /account")

    return '', 200

def send_telegram_message(chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, json=payload)
        app.logger.info(f"Telegram notification sent: {response.json()}")
        return response.json()
    except Exception as e:
        app.logger.error(f"Error sending Telegram notification: {str(e)}")
        return None





@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(request.args.get('next') or url_for('account'))
        else:
            return render_template('login.html', error='Invalid password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))



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
    # context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # context.minimum_version = ssl.TLSVersion.TLSv1_2
    # context.maximum_version = ssl.TLSVersion.TLSv1_3
    # context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384')
    # context.load_cert_chain('cert.pem', 'key.pem')
    webhook_url = f"https://172.232.199.44/{TELEGRAM_BOT_TOKEN}"
    set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={webhook_url}"
    response = requests.get(set_webhook_url)
    app.logger.info(f"Webhook set response: {response.json()}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)),debug=True)