from flask import Flask, request, jsonify, render_template_string,send_file
import numpy as np
import pandas as pd
import io
import os
import csv
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import logging
import traceback
from redis import Redis
from urllib.parse import urlparse
import pickle


app = Flask(__name__)
app.logger.setLevel(logging.INFO)


# Fetch Redis host and port
redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', '6379')

# Check if redis_port is a connection string (contains 'tcp://')
if '://' in redis_port:
    # Parse the connection string
    parsed_url = urlparse(redis_port)
    redis_host = parsed_url.hostname
    redis_port = parsed_url.port
else:
    # Convert the port to an integer if it's just a number
    redis_port = int(redis_port)

# Initialize Redis client
redis_client = Redis(host=redis_host, port=redis_port)

model = None
scaler = None
model_type = 'rf'  # default model type
prepared_data = None

def get_stock_data_from_redis(symbol):
    data = redis_client.get(symbol)
    if data:
        json_data = json.loads(data)
        df = pd.DataFrame.from_dict(json_data, orient='index')
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def prepare_data(df, symbol,sequence_length=10):
    print(f"Preparing data")
    
    if df.empty:
        raise ValueError(f"No data available")

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
        'SPX_Close', 'SMA20', 'SMA50', 'SMA150', 'SMA200',
        'MACD', 'MACD_signal', 'RSI', 'CCI',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_percentage',
        'EMA20', 'HT_TRENDLINE', 'STOCH_K', 'STOCH_D',
        'WILLR', 'AD', 'OBV', 'ATR', 'NATR'
    ]

    # Save the feature list
    redis_client.set(f"{symbol}_features", json.dumps(features))


    print(f"Initial dataframe shape: {df.shape}")
    print(f"Initial dataframe columns: {df.columns}")

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    # Select relevant features
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
        'SPX_Close', 'SMA20', 'SMA50', 'SMA150', 'SMA200',
        'MACD', 'MACD_signal', 'RSI', 'CCI',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_percentage',
        'EMA20', 'HT_TRENDLINE', 'STOCH_K', 'STOCH_D',
        'WILLR', 'AD', 'OBV', 'ATR', 'NATR'
    ]

    # Filter columns
    df = df[features]

    # Handle missing values
    df = df.interpolate().dropna()

    print(f"Dataframe shape after cleaning: {df.shape}")

    # Calculate target (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    print(f"Dataframe shape after adding target: {df.shape}")

    # Prepare features and target
    X = df[features]
    y = df['Target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y.iloc[i+sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(f"Final X shape: {X_seq.shape}")
    print(f"Final y shape: {y_seq.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    print("Finished prepare_data function")
    return (X_train, X_test, y_train, y_test), scaler, df, features



def create_model(model_type, input_shape=None):
    if model_type == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        return LinearRegression()
    elif model_type == 'svr':
        return SVR(kernel='rbf')
    elif model_type == 'xgb':
        return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == 'lstm':
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    else:
        raise ValueError("Invalid model type")

def train_model(X_train, y_train, model_type, symbol, features):
    redis_client.set(f"{symbol}_features", json.dumps(features))
    
    model = create_model(model_type, input_shape=(X_train.shape[1], X_train.shape[2]))
    if model_type == 'lstm':
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    else:
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def predict_next_close(model, scaler, latest_data, model_type, symbol):
    print("in predict next close")
    features = json.loads(redis_client.get(f"{symbol}_features"))
    
    # Convert to DataFrame if it's a numpy array
    if isinstance(latest_data, np.ndarray):
        latest_data = pd.DataFrame(latest_data, columns=features)
    
    # Check for missing features and add them with default values (e.g., 0)
    for feature in features:
        if feature not in latest_data.columns:
            latest_data[feature] = 0
    
    # Ensure the order of features matches the training order
    latest_data = latest_data[features]
    
    latest_scaled = scaler.transform(latest_data)
    if model_type == 'lstm':
        return model.predict(np.array([latest_scaled]))[0][0]
    else:
        return model.predict(latest_scaled.reshape(1, -1))[0]


@app.route('/train', methods=['POST'])
def train():
    global model, scaler, model_type, prepared_data
    print("Received training request")
    print(f"Request JSON: {request.json}")
    
    try:
        if 'symbol' not in request.json:
            print("Error: 'symbol' not found in request JSON")
            return jsonify({"error": "'symbol' is required"}), 400
        
        symbol = request.json['symbol']
        model_type = request.json.get('model_type', 'rf')
        print(f"Training for symbol: {symbol}")
        print(f"Model type: {model_type}")
        df = get_stock_data_from_redis(symbol)
        if df.empty:
            print(f"Error: No data available for symbol {symbol}")
            return jsonify({"error": f"No data available for symbol {symbol}"}), 400
        
        print(f"Data shape for {symbol}: {df.shape}")
        print(f"Data columns: {df.columns}")
        print(f"First few rows of data:\n{df.head()}")
        
        (X_train, X_test, y_train, y_test), scaler, prepared_data, features = prepare_data(df, symbol)
   
        app.logger.info(f"Number of features in training data: {X_train.shape[1]}")
        app.logger.info(f"Features in training data: {list(prepared_data.columns)}")

        print(f"Prepared data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        model = train_model(X_train, y_train, model_type, symbol, features)
        
        if model_type == 'lstm':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        }
        
        feature_importance = None
        if model_type in ['rf', 'xgb']:
            feature_importance = dict(zip(prepared_data.columns, model.feature_importances_))
        
        # Save results to Redis
        redis_client.set(f"{symbol}_model_type", model_type)
        redis_client.set(f"{symbol}_model_metrics", json.dumps(metrics))
        if feature_importance:
            redis_client.set(f"{symbol}_feature_importance", json.dumps(feature_importance))
        
        # Save model and scaler
        redis_client.set(f"{symbol}_model", pickle.dumps(model))
        redis_client.set(f"{symbol}_scaler", pickle.dumps(scaler))
        
        print("Training completed successfully")
        return jsonify({
            "message": f"Model {model_type} trained successfully for {symbol}",
            "metrics": metrics,
            "feature_importance": feature_importance
        }), 200
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'symbol' not in request.json:
            return jsonify({"error": "Symbol is required"}), 400
        
        symbol = request.json['symbol']
        data = request.json['data']
        
        app.logger.info(f"Received prediction request for symbol: {symbol}")
        app.logger.debug(f"Input data: {data}")
        
        model_type = redis_client.get(f"{symbol}_model_type")
        if not model_type:
            return jsonify({"error": "Model not trained for this symbol"}), 400
        
        model_type = model_type.decode('utf-8')
        app.logger.info(f"Model type: {model_type}")
        
        model = pickle.loads(redis_client.get(f"{symbol}_model"))
        scaler = pickle.loads(redis_client.get(f"{symbol}_scaler"))
        
        # Convert the data back to a DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        
        app.logger.debug(f"Processed DataFrame shape: {df.shape}")
        
        # Convert columns back to appropriate types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                # Column couldn't be converted to numeric, leave it as is
                pass        

        features = json.loads(redis_client.get(f"{symbol}_features"))
        app.logger.info(f"Number of features used in training: {len(features)}")
        app.logger.info(f"Features used in training: {features}")
        
        # Ensure all required features are present
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0  # or use a more sophisticated imputation method
        
        df = df[features]
        latest_data = df.tail(10)
        app.logger.info(f"Number of features in prediction data: {latest_data.shape[1]}")
        app.logger.debug(f"Latest data shape: {latest_data.shape}")
        
        prediction = predict_next_close(model, scaler, latest_data, model_type, symbol)
        app.logger.info(f"Prediction successful: {prediction}")
        
        # Calculate confidence interval (for RandomForest)
        if model_type == 'rf':
            predictions = []
            for estimator in model.estimators_:
                if model_type == 'lstm':
                    pred = estimator.predict(np.array([scaler.transform(latest_data)]))[0][0]
                else:
                    pred = estimator.predict(scaler.transform(latest_data).reshape(1, -1))[0]
                predictions.append(pred)
            confidence_interval = stats.t.interval(0.95, len(predictions)-1, loc=np.mean(predictions), scale=stats.sem(predictions))
        else:
            confidence_interval = None
        
        # Generate a short-term forecast (next 5 days)
        forecast = []
        temp_data = latest_data.copy()
        for _ in range(5):
            next_pred = predict_next_close(model, scaler, temp_data, model_type, symbol)
            forecast.append(float(next_pred))
            new_row = temp_data.iloc[-1].copy()
            new_row['Close'] = next_pred
            temp_data = pd.concat([temp_data.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
        
        return jsonify({
            "prediction": float(prediction),
            "confidence_interval": confidence_interval,
            "forecast": forecast
        }), 200
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    

@app.route('/download_prepared_data', methods=['GET'])
def download_prepared_data():
    global prepared_data
    if prepared_data is None:
        return jsonify({"error": "No prepared data available. Train a model first."}), 400
    
    # Create a CSV file in memory
    output = io.StringIO()
    prepared_data.to_csv(output, index=True)
    output.seek(0)
    
    # Send the file
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename='prepared_data.csv'
    )

@app.route('/view_prepared_data', methods=['GET'])
def view_prepared_data():
    global prepared_data
    if prepared_data is None:
        return "No prepared data available. Train a model first."
    
    # Convert the DataFrame to HTML
    html_table = prepared_data.to_html(classes='data')
    
    # Create a simple HTML page with the table
    html_content = f"""
    <html>
        <head>
            <title>Prepared Data</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Prepared Data</h1>
            {html_table}
        </body>
    </html>
    """
    
    return html_content

@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Service Status</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1 { color: #333; }
            .status { font-weight: bold; color: green; }
        </style>
    </head>
    <body>
        <h1>ML Service Status</h1>
        <p>Status: <span class="status">Running</span></p>
        <p>This page confirms that the ML service is up and running.</p>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/<path:path>')
def catch_all(path):
    return jsonify({"message": f"Undefined route: {path}"}), 404

if __name__ == '__main__':
    print("ML Service is starting up...")
    app.run(host='0.0.0.0', port=5002)