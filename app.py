from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --- 1. TRAIN THE MODEL ON STARTUP ---
print("Training EV Model with Vehicle Efficiency...")
np.random.seed(42)
n_samples = 1000  # Increased for slightly better training

battery_size_kwh = np.random.uniform(30, 120, n_samples)
temperature_celsius = np.random.uniform(-10, 40, n_samples)
avg_speed_kmh = np.random.uniform(30, 130, n_samples)
ac_heater_on = np.random.choice([0, 1], n_samples)
efficiency_factor = np.random.uniform(0.7, 1.2, n_samples) # NEW: Aerodynamics & weight

# Base range now accounts for vehicle efficiency
base_range = (battery_size_kwh * 6) * efficiency_factor
temp_penalty = (20 - temperature_celsius)**2 * 0.1 
speed_penalty = (avg_speed_kmh - 60) * 1.2         
ac_penalty = ac_heater_on * 25                     

actual_range = base_range - temp_penalty - speed_penalty - ac_penalty + np.random.normal(0, 10, n_samples)

df = pd.DataFrame({
    'Battery_kWh': battery_size_kwh,
    'Temp_C': temperature_celsius,
    'Speed_kmh': avg_speed_kmh,
    'AC_Heater_On': ac_heater_on,
    'Efficiency_Factor': efficiency_factor # Added to training data
})
y = actual_range 

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df, y)
print("Model Ready!")

# --- 2. WEB ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # The model now expects 5 features instead of 4
    features = pd.DataFrame([[
        float(data['battery']),
        float(data['temp']),
        float(data['speed']),
        int(data['ac']),
        float(data['efficiency'])
    ]], columns=['Battery_kWh', 'Temp_C', 'Speed_kmh', 'AC_Heater_On', 'Efficiency_Factor'])
    
    prediction = model.predict(features)[0]
    
    # Ensure range doesn't drop below 0 in extreme edge cases
    final_range = max(0, prediction)
    
    return jsonify({'prediction': round(final_range, 2)})

if __name__ == '__main__':
    app.run(debug=True)