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
    global prepared_data

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
    app.logger.info(f"Actual features being used: {features}")
    app.logger.info(f"Number of features: {len(features)}")
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
    prepared_data = df
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
    app.logger.info(f"Training {model_type} model for {symbol}")
    app.logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    app.logger.info(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
    
    redis_client.set(f"{symbol}_features", json.dumps(features))
    
    model = create_model(model_type, input_shape=(X_train.shape[1], X_train.shape[2]))
    if model_type == 'xgb':
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        app.logger.info(f"Reshaped X_train for XGB: {X_train_reshaped.shape}")
        model.fit(X_train_reshaped, y_train)
    elif model_type == 'lstm':
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
        latest_scaled = latest_scaled.reshape(1, latest_scaled.shape[0], latest_scaled.shape[1])
        prediction = model.predict(latest_scaled)[0][0]
        # For LSTM, we can use the model's loss as a proxy for confidence
        confidence = 1 / (1 + model.evaluate(latest_scaled, np.array([prediction]), verbose=0)[0])
    elif model_type == 'rf':
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict(latest_scaled.reshape(1, -1))[0]
            predictions.append(pred)
        prediction = np.mean(predictions)
        confidence = 1 - (np.std(predictions) / prediction)  # Normalize by prediction value
    elif model_type == 'xgb':
        prediction = model.predict(latest_scaled.reshape(1, -1))[0]
        # For XGBoost, we can use the built-in predict_proba method if it's a classification task
        # For regression, we'll use a simplification based on the tree's leaf values
        leaf_outputs = model.get_booster().predict(xgb.DMatrix(latest_scaled.reshape(1, -1)), pred_leaf=True)
        confidence = 1 / (1 + np.std(leaf_outputs))
    else:
        prediction = model.predict(latest_scaled.reshape(1, -1))[0]
        # For other models, we'll use a simple placeholder confidence
        confidence = 0.5  # You might want to adjust this or implement a more sophisticated method
    return prediction, confidence




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
        
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
        
        # Convert feature importance to regular Python types
        if model_type in ['rf', 'xgb']:
            feature_importance = {k: float(v) for k, v in zip(prepared_data.columns, model.feature_importances_)}
        else:
            feature_importance = None
        
        # Save results to Redis
        redis_client.set(f"{symbol}_model_type", model_type)
        redis_client.set(f"{symbol}_model_metrics", json.dumps(metrics))
        if feature_importance:
            redis_client.set(f"{symbol}_feature_importance", json.dumps(feature_importance))
        
        # Save model and scaler
        redis_client.set(f"{symbol}_model", pickle.dumps(model))
        redis_client.set(f"{symbol}_scaler", pickle.dumps(scaler))
        app.logger.info(f"Prepared data after training: {prepared_data.shape if prepared_data is not None else 'None'}")

        print("Training completed successfully")
        return jsonify({
            "message": f"Model {model_type} trained successfully for {symbol}",
            "metrics": metrics,
            "feature_importance": feature_importance
        }), 200
    except ValueError as ve:
        app.logger.error(f"ValueError in train: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except TypeError as te:
        app.logger.error(f"TypeError in train: {str(te)}")
        return jsonify({"error": str(te)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in train: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred during training"}), 500

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
        
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        
        app.logger.debug(f"Processed DataFrame shape: {df.shape}")
        
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass        

        features = json.loads(redis_client.get(f"{symbol}_features"))
        
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df[features]
        latest_data = df.tail(10)
        app.logger.info(f"Number of features in prediction data: {latest_data.shape[1]}")
        app.logger.debug(f"Latest data shape: {latest_data.shape}")
        
        prediction, confidence = predict_next_close(model, scaler, latest_data, model_type, symbol)
        app.logger.info(f"Prediction successful: {prediction}, Confidence: {confidence}")
        
        # Generate a short-term forecast (next 5 days)
        forecast = []
        forecast_confidence = []
        temp_data = latest_data.copy()
        for _ in range(5):
            next_pred, next_conf = predict_next_close(model, scaler, temp_data, model_type, symbol)
            forecast.append(float(next_pred))
            forecast_confidence.append(float(next_conf))
            new_row = temp_data.iloc[-1].copy()
            new_row['Close'] = next_pred
            temp_data = pd.concat([temp_data.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
        
        return jsonify({
            "prediction": float(prediction),
            "confidence": float(confidence),
            "forecast": forecast,
            "forecast_confidence": forecast_confidence
        }), 200
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    

@app.route('/download_prepared_data', methods=['GET'])
def download_prepared_data():
    global prepared_data

    # Debug: Check if prepared_data is None
    if prepared_data is None:
        app.logger.error("No prepared data available. Train a model first.")
        return jsonify({"error": "No prepared data available. Train a model first."}), 400

    try:
        # Debug: Log shape and columns of prepared_data
        app.logger.info(f"Prepared data shape: {prepared_data.shape}")
        app.logger.info(f"Prepared data columns: {prepared_data.columns}")

        # Create a CSV file in memory
        output = io.StringIO()
        prepared_data.to_csv(output, index=True)
        output.seek(0)
        
        # Debug: Confirm the CSV is created and log the size of the output
    # Debug: Confirm the CSV is created and log the size of the output
        csv_size = len(output.getvalue())
        app.logger.info(f"CSV size: {csv_size} bytes")

        # Send the in-memory file
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name='prepared_data.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        # Debug: Log the error message and traceback
        app.logger.error(f"Error generating CSV: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error generating CSV: {str(e)}"}), 500


@app.route('/view_prepared_data', methods=['GET'])
def view_prepared_data():
    global prepared_data

    # Debug: Check if prepared_data is None
    if prepared_data is None:
        app.logger.error("No prepared data available. Train a model first.")
        return jsonify({"error": "No prepared data available. Train a model first."}), 400

    try:
        # Debug: Log shape and columns of prepared_data
        app.logger.info(f"Prepared data shape: {prepared_data.shape}")
        app.logger.info(f"Prepared data columns: {prepared_data.columns}")

        # Convert DataFrame to JSON
        data_json = prepared_data.to_dict(orient="records")

        # Debug: Log the size of the JSON response
        json_size = len(data_json)
        app.logger.info(f"Number of records in JSON response: {json_size}")

        # Return the JSON response
        return jsonify(data_json)
    except Exception as e:
        # Debug: Log the error message and traceback
        app.logger.error(f"Error viewing prepared data: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error viewing prepared data: {str(e)}"}), 500


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