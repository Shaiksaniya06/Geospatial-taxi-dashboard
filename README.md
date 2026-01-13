# NYC Yellow Taxi Geospatial Dashboard 🚕

## 📌 Project Overview
This project is an interactive geospatial dashboard built using Python and Dash to analyze NYC Yellow Taxi trip data.  
It visualizes ride hotspots, trip trends, distance distribution, and passenger patterns.

## 🎯 Objectives
- Identify high-demand pickup zones (hotspots)
- Analyze trips over time
- Understand trip distance patterns
- Explore passenger count behavior

## 🛠 Tech Stack
- Python
- Dash & Plotly
- Pandas, NumPy
- Scikit-learn
- NYC Yellow Taxi Dataset (Parquet)

## 📊 Dashboard Features
- 🗺 Mapbox hotspot visualization  
- 📈 Trips over time (line chart)  
- 📊 Top zones / summary chart  
- 📉 Trip distance histogram  
- 🎛 Interactive filters (date, distance, passengers, time of day)

## ▶ How to Run the Project
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
