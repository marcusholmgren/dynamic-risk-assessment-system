import json
import os
import subprocess

import pandas as pd
from flask import Flask, jsonify, request

# import create_prediction_model
from diagnostics import model_predictions, dataframe_summary, dataframe_missing_values, execution_time

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    if not request.is_json:
        return jsonify({"error": "Please provide 'data_file' in request."}), 400
    else:
        try:
            data_file = request.get_json()['data_file']
            if not os.path.isfile(data_file):
                return jsonify({"error": "File not found"}), 404
            else:
                df = pd.read_csv(data_file)
                preds = model_predictions(df)
                return jsonify({"predictions": preds})
        except Exception as e:
            return jsonify({"error": "Please provide 'data_file' in request."}), 400


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    _ = subprocess.run(["python", "scoring.py"], stdout=subprocess.DEVNULL)
    with open(os.path.join(output_model_path, 'latestscore.txt'), 'r') as score_file:
        score = score_file.read()
    return jsonify({"F1": score})



@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    summary = dataframe_summary()
    return jsonify({"stats": summary})



@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    timing = execution_time()
    missing_values = dataframe_missing_values()
    return jsonify({
        "timing_sec": timing,
        "pct_missing_values": missing_values
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
