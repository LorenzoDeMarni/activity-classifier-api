from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from scipy import stats
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
try:
    model = joblib.load("activity_classifier.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Window size parameter
window_size = 500

def extract_features(segment):
    return {
        "mean": np.mean(segment, axis=0),
        "std": np.std(segment, axis=0),
        "min": np.min(segment, axis=0),
        "max": np.max(segment, axis=0),
        "range": np.ptp(segment, axis=0),
        "variance": np.var(segment, axis=0),
        "median": np.median(segment, axis=0),
        "rms": np.sqrt(np.mean(np.square(segment), axis=0)),
        "kurtosis": stats.kurtosis(segment, axis=0),
        "skewness": stats.skew(segment, axis=0),
    }

def features_to_dataframe(features_list):
    feature_names = ['mean', 'std', 'min', 'max', 'range', 'variance', 'median', 'rms', 'kurtosis', 'skewness']
    axes = ['x', 'y', 'z', 'abs']

    columns = [f"{feature}_{axis}" for feature in feature_names for axis in axes]
    data = [[value for feature in feature_names for value in features[feature]] for features in features_list]

    return pd.DataFrame(data, columns=columns)

def segment_data_5s(data, window_size):
    return np.array([data.iloc[i:i + window_size, 1:].values for i in range(0, len(data), window_size) if len(data.iloc[i:i + window_size]) == window_size])

@app.route("/")
def index():
    return jsonify({"message": "Activity Classifier API is running!"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400

    try:
        df = pd.read_csv(file)

        if df.empty or len(df.columns) < 2:
            return jsonify({"error": "CSV file is empty or invalid"}), 400

        segments = segment_data_5s(df, window_size)
        if len(segments) == 0:
            return jsonify({"error": "No valid segments found in data"}), 400

        features_list = [extract_features(seg) for seg in segments]
        feature_df = features_to_dataframe(features_list)
        num_axes = 4  # x, y, z, abs

        # Use modulus to extract features for each axis separately
        # This works because our columns are organized such that:
        # Column 0, 4, 8... are x-axis features
        # Column 1, 5, 9... are y-axis features
        # Column 2, 6, 10... are z-axis features
        # Column 3, 7, 11... are abs-axis features
        x_cols   = [col for i, col in enumerate(feature_df) if i % num_axes == 0]
        y_cols   = [col for i, col in enumerate(feature_df) if i % num_axes == 1]
        z_cols   = [col for i, col in enumerate(feature_df) if i % num_axes == 2]
        abs_cols = [col for i, col in enumerate(feature_df) if i % num_axes == 3]


        # Reorganize final dataset to group features by axis type
        final_dataset = final_dataset[x_cols + y_cols + z_cols + abs_cols + ['activity']]
        predictions = model.predict(final_dataset)

        labels = ["walking" if p == 0 else "jumping" for p in predictions]
        jumping_count, walking_count = np.sum(predictions == 1), np.sum(predictions == 0)
        final_classification = "Jumping" if jumping_count > walking_count else "Walking"

        return jsonify({
            "segments": [{"segment": i, "prediction": label} for i, label in enumerate(labels)],
            "final_classification": final_classification,
            "walking_count": int(walking_count),
            "jumping_count": int(jumping_count),
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "CSV file is empty"}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Invalid CSV format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
