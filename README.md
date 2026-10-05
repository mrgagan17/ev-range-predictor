# ev-range-predictor

An end-to-end Machine Learning web application that predicts realistic Electric Vehicle (EV) range based on environmental factors, driving dynamics, cabin climate control, and vehicle aerodynamics.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📌 Problem Overview

Official EV range ratings (such as EPA or WLTP) are tested under controlled laboratory conditions. In the real world, an electric car's range degrades significantly due to:

- **Extreme Temperatures:** Sub-zero cold slows down lithium-ion chemistry; intense heat increases battery cooling overhead.
- **Highway Speeds:** Aerodynamic drag increases quadratically with speed, draining batteries much faster above 90 km/h.
- **Cabin Climate Control:** Resistive heating in winter can reduce range by up to 30–35%.
- **Vehicle Aerodynamics & Weight:** A boxy, 4-ton SUV consumes far more energy per kilometer than a streamlined sedan.

This project bridges that gap by training an ensemble Machine Learning model to simulate these non-linear physical interactions and deliver accurate, real-world range estimates.

---

## 🚀 Key Features

- **Trained with Ensemble ML (`RandomForestRegressor`):** Captures non-linear degradation curves (temperature penalties, velocity drag) that simple linear regression cannot model.
- **Real Vehicle Presets:** Pre-configured with battery sizes and aerodynamic efficiency factors for:
  - Tesla Model 3 (Standard)
  - Hyundai Ioniq 5 (Long Range)
  - Tata Nexon EV (Long Range)
  - GMC Hummer EV (Heavy SUV)
- **Interactive Dark-Mode Dashboard:** Clean, responsive UI with real-time dynamic inputs and asynchronous prediction updates via `fetch` API.
- **RESTful Flask Backend:** Exposes a JSON `/predict` endpoint that validates input parameters and returns inference results instantly.

---

## 🛠️ Tech Stack

- **Machine Learning & Data:** Python, Scikit-Learn, Pandas, NumPy
- **Backend API:** Flask
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (Fetch API)

---

## 📊 Model & Feature Engineering

The model evaluates **5 key input features**:

| Feature | Unit / Type | Description |
| :--- | :--- | :--- |
| `Battery_kWh` | Float (kWh) | Usable battery pack capacity |
| `Temp_C` | Float (°C) | Ambient outside temperature (-10°C to 40°C) |
| `Speed_kmh` | Float (km/h) | Average cruising speed |
| `AC_Heater_On` | Binary (0 / 1) | Cabin climate control state |
| `Efficiency_Factor` | Float (0.65 – 1.15) | Drag coefficient and curb weight multiplier |

### Why Random Forest Regressor?
Physical battery discharge does not follow a straight line ($y = mx + b$). Freezing conditions and high highway speeds compound each other exponentially. A Random Forest ensemble of 100 decision trees effectively learns these split interactions without overfitting or assuming linear relationships.

---

## 📁 Project Structure

```text
ev-range-predictor/
├── app.py              # Flask server & ML training/inference pipeline
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # Documentation
└── templates/
    └── index.html      # Responsive frontend web dashboard
