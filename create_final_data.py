#!/usr/bin/env python3
"""
Comprehensive Feature Engineering Pipeline for Dengue Outbreak Prediction
Creates final_data.csv with all merged features including weather data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
from collections import defaultdict

# Import free weather data functions
try:
    from fetch_weather_data_free import add_weather_to_dataframe, fetch_world_bank_climate_data
    WEATHER_MODULE_AVAILABLE = True
except ImportError:
    WEATHER_MODULE_AVAILABLE = False
    print("  ⚠ Weather module not found. Using OpenWeatherMap only.")

print("=" * 60)
print("COMPREHENSIVE FEATURE ENGINEERING PIPELINE")
print("Creating final_data.csv with all merged features + Weather Data")
print("=" * 60)

# OpenWeatherMap API Configuration
API_KEY = "809484d70ea1cf1221b724109ab4cad8"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Country coordinates (capital cities) for weather data
COUNTRY_COORDINATES = {
    'BGD': (23.8103, 90.4125),   # Dhaka, Bangladesh
    'THA': (13.7563, 100.5018),  # Bangkok, Thailand
    'IDN': (-6.2088, 106.8456),  # Jakarta, Indonesia
    'IND': (28.6139, 77.2090),   # New Delhi, India
    'LKA': (6.9271, 79.8612),    # Colombo, Sri Lanka
    'MMR': (16.8661, 96.1951),   # Yangon, Myanmar
    'NPL': (27.7172, 85.3240),   # Kathmandu, Nepal
    'MDV': (4.1755, 73.5093),     # Male, Maldives
    'TLS': (-8.5569, 125.5603),  # Dili, East Timor
    'BTN': (27.4728, 89.6390),   # Thimphu, Bhutan
}

def fetch_weather_data(lat, lon, date, api_key):
    """
    Fetch weather data from OpenWeatherMap API
    Note: Free tier doesn't support historical data, so we use current weather
    For historical data, you'd need a paid plan or use alternative sources
    """
    try:
        # For current weather (free tier limitation)
        url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'rainfall': data.get('rain', {}).get('1h', 0) if 'rain' in data else 0
            }
        else:
            return None
    except Exception as e:
        print(f"    ⚠ Error fetching weather: {e}")
        return None

def get_weather_for_country_month(country_code, year, month):
    """
    Get average weather for a country in a specific month
    Since we can't get historical data from free API, we use current weather
    as a proxy (this is a limitation - ideally use historical weather data)
    """
    if country_code not in COUNTRY_COORDINATES:
        return None
    
    lat, lon = COUNTRY_COORDINATES[country_code]
    weather = fetch_weather_data(lat, lon, None, API_KEY)
    
    if weather:
        time.sleep(0.1)  # Rate limiting - free tier: 60 calls/minute
    return weather

# Step 1: Load and prepare base data
print("\n[1/7] Loading and preparing base data...")
df_base = pd.read_csv("dengu_out_break.csv", low_memory=False)

# Convert dates
df_base['calendar_start_date'] = pd.to_datetime(df_base['calendar_start_date'])
df_base['calendar_end_date'] = pd.to_datetime(df_base['calendar_end_date'])

# Create a unique location identifier
df_base['location_id'] = df_base['ISO_A0'].astype(str) + '_' + \
                         df_base['adm_1_name'].fillna('').astype(str) + '_' + \
                         df_base['adm_2_name'].fillna('').astype(str)

# Sort by location and date
df_base = df_base.sort_values(['location_id', 'calendar_start_date']).reset_index(drop=True)

print(f"  ✓ Loaded {len(df_base):,} records")
print(f"  ✓ Date range: {df_base['calendar_start_date'].min()} to {df_base['calendar_start_date'].max()}")
print(f"  ✓ Unique locations: {df_base['location_id'].nunique()}")

# Step 2: Create temporal features
print("\n[2/7] Creating temporal features...")

df_features = df_base.copy()

# Basic temporal features
df_features['year'] = df_features['calendar_start_date'].dt.year
df_features['month'] = df_features['calendar_start_date'].dt.month
df_features['day'] = df_features['calendar_start_date'].dt.day
df_features['day_of_year'] = df_features['calendar_start_date'].dt.dayofyear
df_features['quarter'] = df_features['calendar_start_date'].dt.quarter
df_features['week'] = df_features['calendar_start_date'].dt.isocalendar().week
df_features['day_of_week'] = df_features['calendar_start_date'].dt.dayofweek
df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

# Period length (days between start and end)
df_features['period_length'] = (df_features['calendar_end_date'] - df_features['calendar_start_date']).dt.days + 1

# Seasonal indicators (adjust based on your region - this is for tropical regions)
df_features['is_rainy_season'] = df_features['month'].isin([6, 7, 8, 9, 10]).astype(int)
df_features['is_dry_season'] = (~df_features['month'].isin([6, 7, 8, 9, 10])).astype(int)

# Cyclical encoding for month (sine/cosine)
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)

print(f"  ✓ Created temporal features")

# Step 3: Create lag features (cases from previous periods)
print("\n[3/7] Creating lag features...")

# Group by location for lag calculations
lag_periods = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90, 180, 365]

for lag in lag_periods:
    df_features[f'cases_lag_{lag}d'] = df_features.groupby('location_id')['dengue_total'].shift(lag)
    # Fill NaN with 0 for locations without enough history
    df_features[f'cases_lag_{lag}d'] = df_features[f'cases_lag_{lag}d'].fillna(0)

print(f"  ✓ Created {len(lag_periods)} lag features")

# Step 4: Create rolling statistics
print("\n[4/7] Creating rolling statistics...")

rolling_windows = [7, 14, 21, 30, 60, 90, 180, 365]

for window in rolling_windows:
    # Rolling mean
    df_features[f'cases_rolling_mean_{window}d'] = df_features.groupby('location_id')['dengue_total'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    # Rolling sum
    df_features[f'cases_rolling_sum_{window}d'] = df_features.groupby('location_id')['dengue_total'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    )
    
    # Rolling max
    df_features[f'cases_rolling_max_{window}d'] = df_features.groupby('location_id')['dengue_total'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
    )
    
    # Rolling std (variability)
    df_features[f'cases_rolling_std_{window}d'] = df_features.groupby('location_id')['dengue_total'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0)
    )

print(f"  ✓ Created {len(rolling_windows) * 4} rolling statistics features")

# Step 5: Create trend and change features
print("\n[5/7] Creating trend and change features...")

# Week-over-week change
df_features['cases_wow_change'] = df_features.groupby('location_id')['dengue_total'].pct_change(periods=1).fillna(0)
df_features['cases_wow_abs_change'] = df_features.groupby('location_id')['dengue_total'].diff(periods=1).fillna(0)

# Compare recent periods
df_features['cases_7d_vs_30d'] = (df_features['cases_rolling_mean_7d'] - df_features['cases_rolling_mean_30d']).fillna(0)
df_features['cases_7d_vs_30d_pct'] = (
    (df_features['cases_rolling_mean_7d'] - df_features['cases_rolling_mean_30d']) / 
    (df_features['cases_rolling_mean_30d'] + 1) * 100
).fillna(0)

df_features['cases_30d_vs_90d'] = (df_features['cases_rolling_mean_30d'] - df_features['cases_rolling_mean_90d']).fillna(0)
df_features['cases_30d_vs_90d_pct'] = (
    (df_features['cases_rolling_mean_30d'] - df_features['cases_rolling_mean_90d']) / 
    (df_features['cases_rolling_mean_90d'] + 1) * 100
).fillna(0)

# Acceleration (rate of change of cases)
df_features['cases_acceleration'] = df_features.groupby('location_id')['dengue_total'].diff(periods=2).fillna(0)

print(f"  ✓ Created trend and change features")

# Step 6: Create historical seasonal features
print("\n[6/7] Creating historical seasonal features...")

# Historical average for same month (across all years)
df_features['historical_monthly_avg'] = df_features.groupby(['location_id', 'month'])['dengue_total'].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(0)

# Historical average for same quarter
df_features['historical_quarterly_avg'] = df_features.groupby(['location_id', 'quarter'])['dengue_total'].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(0)

# Historical max for same month
df_features['historical_monthly_max'] = df_features.groupby(['location_id', 'month'])['dengue_total'].transform(
    lambda x: x.shift(1).expanding().max()
).fillna(0)

# Cases in same month previous year
df_features['cases_same_month_prev_year'] = df_features.groupby('location_id')['dengue_total'].shift(365).fillna(0)

# Cases in same quarter previous year (approximate)
df_features['cases_same_quarter_prev_year'] = df_features.groupby('location_id')['dengue_total'].shift(365).fillna(0)

# Ratio to historical average
df_features['cases_to_historical_monthly_avg'] = (
    df_features['dengue_total'] / (df_features['historical_monthly_avg'] + 1)
).fillna(0)

print(f"  ✓ Created historical seasonal features")

# Step 7: Create location-based aggregated features
print("\n[7/7] Creating location-based aggregated features...")

# Total cases in location (cumulative)
df_features['location_total_cases'] = df_features.groupby('location_id')['dengue_total'].cumsum()

# Average cases per period for location
df_features['location_avg_cases'] = df_features.groupby('location_id')['dengue_total'].transform(
    lambda x: x.expanding().mean()
)

# Location outbreak intensity (cases per period length)
df_features['cases_per_day'] = df_features['dengue_total'] / (df_features['period_length'] + 1)

# Region-level features
df_features['region_total_cases'] = df_features.groupby(['region', 'calendar_start_date'])['dengue_total'].transform('sum')
df_features['region_avg_cases'] = df_features.groupby('region')['dengue_total'].transform('mean')

# Country-level features
df_features['country_total_cases'] = df_features.groupby(['ISO_A0', 'calendar_start_date'])['dengue_total'].transform('sum')
df_features['country_avg_cases'] = df_features.groupby('ISO_A0')['dengue_total'].transform('mean')

print(f"  ✓ Created location-based aggregated features")

# Step 8: Add Weather Data (Using FREE Sources)
print("\n[8/9] Adding weather data from FREE sources...")
print("  Options:")
print("    1. World Bank Climate API (FREE, no key needed) - RECOMMENDED")
print("    2. OpenWeatherMap (current weather only - free tier limitation)")
print("    3. Seasonal estimates (fallback)")

# Try to use free weather data module first
if WEATHER_MODULE_AVAILABLE:
    print("\n  Using World Bank Climate API (FREE)...")
    try:
        df_features = add_weather_to_dataframe(
            df_features,
            country_col='ISO_A0',
            date_col='calendar_start_date',
            weatherapi_key=None,  # Optional: add WeatherAPI.com key if you have one
            noaa_token=None  # Optional: add NOAA token if you have one
        )
        print("  ✓ Weather data added using World Bank API!")
    except Exception as e:
        print(f"  ⚠ Error with World Bank API: {e}")
        print("  Falling back to OpenWeatherMap...")
        WEATHER_MODULE_AVAILABLE = False

# Fallback to OpenWeatherMap if World Bank fails
if not WEATHER_MODULE_AVAILABLE or 'temperature' not in df_features.columns:
    print("\n  Using OpenWeatherMap API (current weather only)...")
    
    # Initialize weather columns
    df_features['temperature'] = np.nan
    df_features['temp_min'] = np.nan
    df_features['temp_max'] = np.nan
    df_features['humidity'] = np.nan
    df_features['pressure'] = np.nan
    df_features['wind_speed'] = np.nan
    df_features['rainfall'] = np.nan
    
    # Get unique country-month combinations to minimize API calls
    unique_countries = df_features['ISO_A0'].unique()
    weather_cache = {}
    
    for country in unique_countries:
        if country in COUNTRY_COORDINATES:
            print(f"  Fetching weather for {country}...")
            weather = get_weather_for_country_month(country, None, None)
            if weather:
                weather_cache[country] = weather
                time.sleep(0.2)
    
    # Fill weather data
    for country in unique_countries:
        if country in weather_cache:
            weather = weather_cache[country]
            mask = df_features['ISO_A0'] == country
            df_features.loc[mask, 'temperature'] = weather['temperature']
            df_features.loc[mask, 'temp_min'] = weather['temp_min']
            df_features.loc[mask, 'temp_max'] = weather['temp_max']
            df_features.loc[mask, 'humidity'] = weather['humidity']
            df_features.loc[mask, 'pressure'] = weather['pressure']
            df_features.loc[mask, 'wind_speed'] = weather['wind_speed']
            df_features.loc[mask, 'rainfall'] = weather['rainfall']
        else:
            # Default values
            mask = df_features['ISO_A0'] == country
            df_features.loc[mask & df_features['temperature'].isna(), 'temperature'] = 25.0
            df_features.loc[mask & df_features['humidity'].isna(), 'humidity'] = 70.0
            df_features.loc[mask & df_features['rainfall'].isna(), 'rainfall'] = 0.0

# Create derived weather features
if 'temperature' in df_features.columns:
    df_features['temp_optimal'] = ((df_features['temperature'] >= 20) & (df_features['temperature'] <= 30)).astype(int)
    df_features['temp_too_cold'] = (df_features['temperature'] < 20).astype(int)
    df_features['temp_too_hot'] = (df_features['temperature'] > 30).astype(int)
    df_features['humidity_high'] = (df_features['humidity'] > 70).astype(int)

print(f"  ✓ Weather features added (temperature, humidity, rainfall, wind_speed)")

# Step 9: Feature Selection - Keep Only Important Features
print("\n[9/9] Feature Selection - Keeping only important features...")

# Define features to keep (important for prediction)
features_to_keep = [
    # Identifiers and target
    'ISO_A0', 'calendar_start_date', 'dengue_total', 'region',
    
    # Temporal (important)
    'year', 'month', 'quarter', 'day_of_year', 'is_rainy_season',
    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'period_length',
    
    # Lag features (key periods only)
    'cases_lag_1d', 'cases_lag_7d', 'cases_lag_14d', 'cases_lag_30d', 'cases_lag_90d', 'cases_lag_365d',
    
    # Rolling statistics (mean and std only, key windows)
    'cases_rolling_mean_7d', 'cases_rolling_std_7d',
    'cases_rolling_mean_14d', 'cases_rolling_std_14d',
    'cases_rolling_mean_30d', 'cases_rolling_std_30d',
    'cases_rolling_mean_90d', 'cases_rolling_std_90d',
    
    # Trend features
    'cases_wow_change', 'cases_wow_abs_change',
    'cases_7d_vs_30d', 'cases_7d_vs_30d_pct',
    'cases_30d_vs_90d', 'cases_30d_vs_90d_pct',
    'cases_acceleration',
    
    # Historical seasonal
    'historical_monthly_avg', 'historical_monthly_max',
    'cases_same_month_prev_year',
    'cases_to_historical_monthly_avg',
    
    # Location aggregated (key ones)
    'location_avg_cases', 'cases_per_day',
    'country_total_cases', 'country_avg_cases',
    
    # Weather features (NEW!)
    'temperature', 'temp_min', 'temp_max', 'humidity', 'pressure',
    'wind_speed', 'rainfall',
    'temp_optimal', 'temp_too_cold', 'temp_too_hot', 'humidity_high',
]

# Keep only features that exist in dataframe
features_to_keep = [f for f in features_to_keep if f in df_features.columns]

# Remove unnecessary features
features_to_remove = [col for col in df_features.columns if col not in features_to_keep]

print(f"  Features to keep: {len(features_to_keep)}")
print(f"  Features to remove: {len(features_to_remove)}")

# Create cleaned dataset
df_cleaned = df_features[features_to_keep].copy()

print(f"  ✓ Cleaned dataset: {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} features")

# Step 10: Final cleanup and save
print("\n" + "=" * 60)
print("Finalizing dataset...")

# Remove rows with all NaN in key features (if any)
initial_rows = len(df_features)
df_features = df_features.dropna(subset=['dengue_total', 'calendar_start_date'])
final_rows = len(df_features)

if initial_rows != final_rows:
    print(f"  ⚠ Removed {initial_rows - final_rows} rows with missing critical data")

# Fill remaining NaN values with 0 for numeric features (except target)
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'dengue_total':  # Don't fill target
        df_cleaned[col] = df_cleaned[col].fillna(0)

# Sort final dataset
df_cleaned = df_cleaned.sort_values(['ISO_A0', 'calendar_start_date']).reset_index(drop=True)

# Display summary
print(f"\n✓ Final cleaned dataset shape: {df_cleaned.shape}")
print(f"✓ Total features: {len(df_cleaned.columns)}")
print(f"✓ Total records: {len(df_cleaned):,}")

print("\n" + "=" * 60)
print("Feature categories in cleaned dataset:")
print(f"  - Temporal features: {len([c for c in df_cleaned.columns if c in ['year', 'month', 'quarter', 'day_of_year', 'is_rainy_season', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'period_length']])}")
print(f"  - Lag features: {len([c for c in df_cleaned.columns if 'lag' in c])}")
print(f"  - Rolling statistics: {len([c for c in df_cleaned.columns if 'rolling' in c])}")
print(f"  - Trend features: {len([c for c in df_cleaned.columns if any(x in c for x in ['wow', 'vs_', 'acceleration'])])}")
print(f"  - Historical seasonal: {len([c for c in df_cleaned.columns if 'historical' in c or 'prev_year' in c])}")
print(f"  - Location aggregated: {len([c for c in df_cleaned.columns if any(x in c for x in ['location_', 'country_', 'cases_per_day'])])}")
print(f"  - Weather features: {len([c for c in df_cleaned.columns if any(x in c for x in ['temp', 'humid', 'rain', 'wind', 'pressure'])])}")

# Save to CSV
output_file = "final_data.csv"
print(f"\n💾 Saving to {output_file}...")
df_cleaned.to_csv(output_file, index=False)
print(f"✅ Successfully saved {len(df_cleaned):,} records with {len(df_cleaned.columns)} features to {output_file}")

# Display first few rows
print("\n" + "=" * 60)
print("Sample of final cleaned dataset:")
display_cols = ['ISO_A0', 'calendar_start_date', 'dengue_total', 
                'cases_lag_7d', 'cases_rolling_mean_7d', 'cases_rolling_mean_30d', 
                'month', 'temperature', 'humidity', 'rainfall']
display_cols = [c for c in display_cols if c in df_cleaned.columns]
print(df_cleaned[display_cols].head(10))

# Display feature summary
print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)

print(f"\n📊 All Features in final_data.csv:")
print(f"\nTotal columns: {len(df_cleaned.columns)}")

# Group features by category
feature_categories = {
    'Identifiers & Target': [col for col in df_cleaned.columns if col in ['ISO_A0', 'calendar_start_date', 'dengue_total', 'region']],
    'Temporal': [col for col in df_cleaned.columns if any(x in col for x in ['year', 'month', 'quarter', 'day_of_year', 'season', 'sin', 'cos', 'period_length'])],
    'Lag Features': [col for col in df_cleaned.columns if 'lag' in col],
    'Rolling Statistics': [col for col in df_cleaned.columns if 'rolling' in col],
    'Trend/Change': [col for col in df_cleaned.columns if any(x in col for x in ['wow', 'vs_', 'acceleration'])],
    'Historical Seasonal': [col for col in df_cleaned.columns if 'historical' in col or 'prev_year' in col],
    'Location Aggregated': [col for col in df_cleaned.columns if any(x in col for x in ['location_', 'country_', 'cases_per_day'])],
    'Weather Features': [col for col in df_cleaned.columns if any(x in col for x in ['temp', 'humid', 'rain', 'wind', 'pressure'])]
}

for category, features in feature_categories.items():
    if features:
        print(f"\n{category} ({len(features)} features):")
        for feat in sorted(features)[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(features) > 10:
            print(f"  ... and {len(features) - 10} more")

print("\n" + "=" * 60)
print("✅ Feature engineering complete!")
print("=" * 60)

