# train_transformer_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add

# ------------------------- Load & Preprocess Data -------------------------
print("Loading and preprocessing data...")

data_path = 'data/daily_avg_formatted_final.csv'
df = pd.read_csv(data_path)
print(df.head())
print("Total null values : ")
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date', inplace=True)
df = df.interpolate()

scaler = QuantileTransformer(output_distribution='normal')
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# with open("model/scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# ------------------------- Data Augmentation -------------------------
def create_overlapping_sequences(data, seq_length, step=3):
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        if i + seq_length < len(data):
            X.append(data.iloc[i:i + seq_length].values)
            y.append(data.iloc[i + seq_length].values)
    return np.array(X), np.array(y)

seq_length = 90
X, y = create_overlapping_sequences(df_scaled, seq_length)

def add_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

X_noisy = add_noise(X)
X_final = np.vstack((X, X_noisy))
y_final = np.vstack((y, y))

train_size = int(len(X_final) * 0.85)
X_train, X_test = X_final[:train_size], X_final[train_size:]
y_train, y_test = y_final[:train_size], y_final[train_size:]

# ------------------------- Transformer Model -------------------------
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add, Flatten, Reshape

def transformer_block(inputs, head_size=64, num_heads=4, ff_dim=4, dropout=0.1):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = Add()([attention, inputs])
    attention = LayerNormalization(epsilon=1e-6)(attention)

    feed_forward = Dense(ff_dim, activation="relu")(attention)
    feed_forward = Dropout(dropout)(feed_forward)
    feed_forward = Dense(inputs.shape[-1])(feed_forward)
    feed_forward = Add()([feed_forward, attention])
    return LayerNormalization(epsilon=1e-6)(feed_forward)

inputs = Input(shape=(seq_length, X_train.shape[-1]))
x = transformer_block(inputs)
x = transformer_block(x)
x = transformer_block(x)
# Reshape the output to match the target shape
x = Flatten()(x)  # Flatten the sequence before the Dense layer
x = Dense(y_train.shape[-1], activation='linear')(x) # Output layer with the correct number of features

model = Model(inputs, x)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), loss='mse')

# ------------------------- Training -------------------------
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.save("backend/model/transformer_forecast_model.keras")
print("Model training complete and saved!")

# ------------------------- Evaluation -------------------------
print("Evaluating model...")
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, df.shape[1]))
y_pred_inv = scaler.inverse_transform(model.predict(X_test).reshape(-1, df.shape[1]))

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print(f'\nEvaluation Results:')
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}')

def categorize_aqi(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Satisfactory"
    elif value <= 200:
        return "Moderate"
    elif value <= 300:
        return "Poor"
    elif value <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Forecasting
forecast_steps = 152
predictions = []
input_seq = X_test[-1].copy()

# Create a dictionary to hold the forecasted values for each city
forecast_results = {}

# List of cities
cities = ['Peenya', 'SilkBoard', 'Hombegowda', 'BapujiNagar']

# Loop through each city and generate forecast for its pollutants
for city in cities:
    print(f"Forecasting for {city}...")

    # Extract the relevant columns for the city (e.g., PM2.5_Peenya, NO2_Peenya, etc.)
    city_columns = [col for col in df.columns if city in col]

    # Initialize city forecast dataframe
    city_forecast = pd.DataFrame(columns=city_columns, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps))

    # Forecast for the pollutants of the current city
    city_predictions = []
    input_seq_city = input_seq.copy()

    for _ in range(forecast_steps):
        pred = model.predict(input_seq_city.reshape(1, seq_length, -1))
        city_predictions.append(pred.flatten())
        input_seq_city = np.roll(input_seq_city, -1, axis=0)
        input_seq_city[-1] = pred.flatten()

    city_predictions = scaler.inverse_transform(np.array(city_predictions))

    # Select the columns corresponding to the current city from the predictions
    city_predictions = city_predictions[:, [df.columns.get_loc(col) for col in city_columns]]

    # Store the forecasted values in the dataframe
    city_forecast[city_columns] = city_predictions

    # Apply AQI categorization for each pollutant in the city
    aqi_columns = [f"AQI_{city}"]  # Example: 'AQI_Peenya', 'AQI_SilkBoard', etc.
    for col in aqi_columns:
        if col in city_forecast.columns:
            city_forecast[f'{col}_Category'] = city_forecast[col].apply(categorize_aqi)


    # Store the forecast for the city
    forecast_results[city] = city_forecast
    print(f"Forecasted Values for {city}:")
    print(city_forecast)
    print("\n" + "-"*50)

# Combine forecasts horizontally by aligning on index (date)
combined_df = None

for city, df in forecast_results.items():
    # Rename columns to include city name as suffix
    new_cols = {col: col.replace(f"_{city}", "") for col in df.columns}
    df.rename(columns=new_cols, inplace=True)

    # Add correct city suffix
    df_renamed = df.add_suffix(f"_{city}")
    
    if combined_df is None:
        combined_df = df_renamed
    else:
        combined_df = combined_df.merge(df_renamed, left_index=True, right_index=True)

# Reset index to make 'Date' a column
combined_df.reset_index(inplace=True)
combined_df.rename(columns={'index': 'Date'}, inplace=True)

# Save combined forecast to CSV
combined_df.to_csv("backend/model/forecast_combined.csv", index=False)
print("✅ Combined forecast saved to model/forecast_combined.csv")


    # Combine all city forecasts into a single dataframe
# forecast_combined = pd.concat(
#     [df.assign(City=city) for city, df in forecast_results.items()]
# )

# # Add 'Date' column from index
# forecast_combined.reset_index(inplace=True)
# forecast_combined.rename(columns={'index': 'Date'}, inplace=True)

# # Save to CSV
# forecast_combined.to_csv("model/forecast_combined.csv", index=False)


import joblib
joblib.dump(scaler, "backend/model/scaler.pkl")

