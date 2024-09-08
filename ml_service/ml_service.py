from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
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

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

model = None
scaler = None
model_type = 'rf'  # default model type
def prepare_data(df, sequence_length=10):
    print("Starting prepare_data function")
    print(f"Initial dataframe shape: {df.shape}")
    print(f"Initial dataframe dtypes:\n{df.dtypes}")
    
    # Ensure all required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI', 'MACD']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataframe")
    
    # Convert columns to numeric, replacing non-numeric values with NaN
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    print(f"Dataframe shape after dropping NaN: {df.shape}")
    
    # Calculate target (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()  # Drop the last row which will have NaN target
    print(f"Dataframe shape after calculating target: {df.shape}")
    
    # Prepare features and target
    X = df[required_columns]
    y = df['Target']
    
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
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
    return (X_train, X_test, y_train, y_test), scaler


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

def train_model(X_train, y_train, model_type):
    model = create_model(model_type, input_shape=(X_train.shape[1], X_train.shape[2]))
    if model_type == 'lstm':
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    else:
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

def predict_next_close(model, scaler, latest_data, model_type):
    if model_type == 'lstm':
        latest_scaled = scaler.transform(latest_data)
        return model.predict(np.array([latest_scaled]))[0][0]
    else:
        latest_scaled = scaler.transform(latest_data)
        return model.predict(latest_scaled.reshape(1, -1))[0]


@app.route('/train', methods=['POST'])
def train():
    global model, scaler, model_type
    print("Received training request")
    try:
        stock_data_dict = request.json['data']
        model_type = request.json.get('model_type', 'rf')
        print(f"Received data for {len(stock_data_dict)} stocks")
        print(f"Model type: {model_type}")
        
        all_data = []
        for symbol, data in stock_data_dict.items():
            df = pd.DataFrame.from_dict(data, orient='index')
            df['Symbol'] = symbol
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        print(f"Combined data dtypes:\n{combined_df.dtypes}")
        
        try:
            (X_train, X_test, y_train, y_test), scaler = prepare_data(combined_df)
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        model = train_model(X_train, y_train, model_type)
        
        # Make predictions on test set
        if model_type == 'lstm':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        
        print(f"y_test dtype: {y_test.dtype}")
        print(f"y_pred dtype: {y_pred.dtype}")
        
        # Calculate metrics
        try:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            raise
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        }
        
        # Get feature importance if available
        feature_importance = None
        if model_type in ['rf', 'xgb']:
            feature_importance = model.feature_importances_.tolist()
        
        print("Training completed successfully")
        return jsonify({
            "message": f"Model {model_type} trained successfully",
            "metrics": metrics,
            "feature_importance": feature_importance
        }), 200
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, model_type
    if model is None:
        return jsonify({"error": "Model not trained"}), 400
    
    data = request.json['data']
    model_type = request.json.get('model_type', 'rf')
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.sort_index(ascending=True)  # Ensure data is in chronological order
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'SMA50', 'RSI', 'MACD']
    latest_data = df[features].tail(10)  # Get last 10 days of data
    
    prediction = predict_next_close(model, scaler, latest_data, model_type)
    
    # Calculate confidence interval (example for RandomForest)
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
        next_pred = predict_next_close(model, scaler, temp_data, model_type)
        forecast.append(float(next_pred))
        new_row = temp_data.iloc[-1].copy()
        new_row['Close'] = next_pred
        temp_data = temp_data.append(new_row, ignore_index=True)
        temp_data = temp_data.iloc[1:]  # Remove the oldest day
    
    return jsonify({
        "prediction": float(prediction),
        "confidence_interval": confidence_interval,
        "forecast": forecast
    }), 200

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
    app.run(host='0.0.0.0', port=5002)