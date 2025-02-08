import os
import pandas as pd
import numpy as np
import pickle

# Define paths relative to the project root
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'travel_budget_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
PIPELINE_PATH = os.path.join(MODEL_DIR, 'pipeline.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Load the trained model, pipeline, and feature columns
try:
    best_model = load_pickle(MODEL_PATH)
    pipeline = load_pickle(PIPELINE_PATH)
    feature_columns = load_pickle(FEATURE_COLUMNS_PATH)
except Exception as e:
    raise Exception(f"Error loading model files: {str(e)}")

def calculate_distance(from_place, to_place):
    distances = {
        ('Bengaluru', 'Mysuru'): 160,
        ('Hubballi', 'Dharwad'): 20,
        ('Mangaluru', 'Udupi'): 60,
        # ... (keep all the distance pairs from original code)
        ('Kodagu', 'Mandya'): 130,
    }
    key = (from_place, to_place)
    reverse_key = (to_place, from_place)
    return distances.get(key, distances.get(reverse_key, 1))

def calculate_base_travel_cost(mode, distance, travelers):
    if travelers <= 0:
        travelers = 1

    base_rates = {
        'Car': {'base': 800, 'per_km': 12, 'maintenance': 0.5},
        'Bus': {'base': 300, 'per_km': 4, 'service': 0.3},
        'Train': {'base': 500, 'per_km': 6, 'reservation': 0.4},
        'Flight': {'base': 2000, 'per_km': 20, 'airport_charges': 0.8}
    }

    mode_costs = base_rates[mode]
    base_cost = (mode_costs['base'] +
                 (mode_costs['per_km'] * distance) *
                 (1 + mode_costs.get(list(mode_costs.keys())[2], 0)))

    if mode == 'Car':
        return base_cost / min(max(travelers, 1), 4)
    else:
        return base_cost

def create_features(input_data):
    df_processed = input_data.copy()
    df_processed['distance_km'] = df_processed.apply(
        lambda row: calculate_distance(row['from_place'], row['to_place']), axis=1)
    df_processed['number_of_travelers'] = df_processed['number_of_travelers'].clip(lower=1)
    df_processed['total_travel_cost'] = df_processed.apply(
        lambda row: calculate_base_travel_cost(row['trip_mode'], row['distance_km'], row['number_of_travelers']), axis=1)
    df_processed['total_food_cost'] = df_processed.apply(
        lambda row: max(1, np.ceil(row['distance_km'] / 300)) * max(row['number_of_travelers'], 1) * 400 * 
        (1.2 if row['trip_mode'] == 'Flight' else 1.0), axis=1)
    
    mode_dummies = pd.get_dummies(df_processed['trip_mode'], prefix='mode')
    df_processed = pd.concat([df_processed, mode_dummies], axis=1)
    
    df_processed['cost_per_km'] = df_processed['total_travel_cost'] / df_processed['distance_km'].replace(0, 1)
    df_processed['cost_per_person'] = df_processed['total_travel_cost'] / df_processed['number_of_travelers']
    df_processed['distance_per_traveler'] = df_processed['distance_km'] / df_processed['number_of_travelers']
    df_processed['total_cost'] = df_processed['total_travel_cost'] + df_processed['total_food_cost']
    df_processed['distance_squared'] = df_processed['distance_km'] ** 2
    df_processed['travelers_squared'] = df_processed['number_of_travelers'] ** 2
    df_processed['distance_travelers'] = df_processed['distance_km'] * df_processed['number_of_travelers']
    df_processed['log_distance'] = np.log1p(df_processed['distance_km'])
    df_processed['log_total_cost'] = np.log1p(df_processed['total_cost'])
    df_processed['food_cost_ratio'] = df_processed['total_food_cost'] / df_processed['total_cost']
    df_processed['travel_cost_ratio'] = df_processed['total_travel_cost'] / df_processed['total_cost']
    
    df_processed = df_processed.drop(['trip_mode', 'from_place', 'to_place'], axis=1)

    # Ensure all expected features exist
    for col in feature_columns:
        if col not in df_processed:
            df_processed[col] = 0

    return df_processed[feature_columns]

def predict_budget(input_data):
    try:
        # Convert input data to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        features = create_features(input_data)
        features_transformed = pipeline.transform(features)
        predictions = best_model.predict(features_transformed)
        return predictions[0]
    except Exception as e:
        raise Exception(f"Error predicting budget: {str(e)}")