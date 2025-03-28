from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from scipy import stats
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = joblib.load("activity_classifier.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Window size parameter
window_size = 500

def extract_features(segment):
    features = {}
    features["mean"] = np.mean(segment, axis=0)
    features["std"] = np.std(segment, axis=0)
    features["min"] = np.min(segment, axis=0)
    features["max"] = np.max(segment, axis=0)
    features["range"] = np.ptp(segment, axis=0)
    features["variance"] = np.var(segment, axis=0)
    features["median"] = np.median(segment, axis=0)
    features["rms"] = np.sqrt(np.mean(np.square(segment), axis=0))
    features["kurtosis"] = stats.kurtosis(segment, axis=0)
    features["skewness"] = stats.skew(segment, axis=0)
    return features

def features_to_dataframe(features_list):
    feature_names = ['mean', 'std', 'min', 'max', 'range', 'variance', 'median', 'rms', 'kurtosis', 'skewness']
    axes = ['x', 'y', 'z', 'abs']
    
    columns = []
    for feature in feature_names:
        for axis in axes:
            columns.append(f"{feature}_{axis}")
    
    data = []
    for features in features_list:
        row = []
        for feature in feature_names:
            row.extend(features[feature])
        data.append(row)
    
    return pd.DataFrame(data, columns=columns)

def segment_data_5s(data, window_size):
    segments = []
    for i in range(0, len(data), window_size):
        segment = data.iloc[i:i + window_size, 1:].values
        if len(segment) == window_size:
            segments.append(segment)
    return np.array(segments)

@app.route('/')
def index():
    return render_template('ml_activity_classifier.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
            
        if len(df.columns) < 2:
            return jsonify({'error': 'CSV file must contain at least 2 columns'}), 400
        
        # Process the data
        segments = segment_data_5s(df, window_size)
        if len(segments) == 0:
            return jsonify({'error': 'No valid segments found in data'}), 400
            
        features_list = []
        for seg in segments:
            features = extract_features(seg)
            features_list.append(features)
        
        feature_df = features_to_dataframe(features_list)
        predictions = model.predict(feature_df)
        
        # Convert predictions to labels
        labels = ["walking" if p == 0 else "jumping" for p in predictions]
        
        # Calculate final classification
        jumping_count = (predictions == 1).sum()
        walking_count = (predictions == 0).sum()
        final_classification = 'Jumping' if jumping_count > walking_count else 'Walking'
        
        # Prepare results
        results = {
            'segments': [{'segment': i, 'prediction': label} for i, label in enumerate(labels)],
            'final_classification': final_classification,
            'walking_count': int(walking_count),
            'jumping_count': int(jumping_count)
        }
        
        return jsonify(results)
    
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'CSV file is empty'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Invalid CSV format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 