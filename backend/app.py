from flask import Flask, request, jsonify
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load forecast data
forecast_path = os.path.join(os.path.join("model" ,'forecast_combined.csv'))
forecast_df = pd.read_csv(forecast_path)

@app.route('/api/forecast', methods=['POST'])
def get_forecast_by_city():
    data = request.get_json()
    city = data.get('city')
    date = data.get('date')
    if not city or not date:
        return jsonify({"error": "Missing city or date"}), 400

    # Filter for the given date
    match = forecast_df[forecast_df['Date'] == date]
    if match.empty:
        return jsonify({"forecast": {}})

    # Extract all columns for this city
    city_cols = [col for col in forecast_df.columns if col.endswith(f"_{city}")]
    forecast_values = match.iloc[0][city_cols].to_dict()
    forecast_clean = {}
    for key, val in forecast_values.items():
        if "Temp" in key or "RH" in key:
            continue  
        if "Category" in key:
            forecast_clean["AQI_Category"] = val
        else:
            base_key = key.split("_")[0]
            forecast_clean[base_key] = val
    return jsonify({"forecast": forecast_clean})

if __name__ == '__main__':
    app.run(debug=True)