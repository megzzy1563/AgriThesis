import pandas as pd
import numpy as np

def categorize_npk_maize(value, nutrient):
    if nutrient == 'N':
        # Maize is a heavy nitrogen feeder
        if value < 80:  # Maize needs higher N than many crops
            return 'Low'
        elif value < 140:
            return 'Medium'
        else:
            return 'High'
    elif nutrient == 'P':
        if value < 15:  # Maize is moderately demanding in P
            return 'Low'
        elif value < 30:
            return 'Medium'
        else:
            return 'High'
    elif nutrient == 'K':
        if value < 80:  # Maize requires good K levels for stalk strength
            return 'Low'
        elif value < 150:
            return 'Medium'
        else:
            return 'High'

def categorize_ph_maize(ph_value):
    if ph_value < 5.5:
        return 'Very Acidic'
    elif ph_value < 6.0:
        return 'Acidic'
    elif ph_value <= 7.0:
        return 'Optimal'  # Maize prefers slightly acidic to neutral pH
    elif ph_value <= 7.5:
        return 'Slightly Alkaline'
    else:
        return 'Alkaline'

def categorize_rainfall_maize(rainfall_value):
    if rainfall_value < 500:  # mm per growing season
        return 'Insufficient'
    elif rainfall_value < 750:
        return 'Marginal'
    elif rainfall_value <= 1200:
        return 'Optimal'  # Maize typically needs 500-1200mm of rainfall
    else:
        return 'Excessive'

def rainfall_adequacy(rainfall):
    if rainfall < 500:
        return (rainfall / 500)  # Less than optimal, scaled 0-1
    elif rainfall <= 1200:
        return 1.0  # Optimal range
    else:
        return 1 - min(1, (rainfall - 1200) / 800)  # Penalize excessive rainfall

def get_application_method(rainfall_status, pH_status):
    """Determine the application method based on conditions"""
    if rainfall_status == "Insufficient":
        return "Split Application - Apply in small doses with irrigation"
    elif rainfall_status == "Excessive":
        return "Use Slow-Release Formulation to prevent leaching"
    elif pH_status == "Very Acidic" or pH_status == "Acidic":
        return "Apply after lime treatment for best results"
    elif pH_status == "Alkaline":
        return "Apply with sulfur amendments for best uptake"
    else:
        return "Standard application methods recommended"

def prepare_sensor_data_for_prediction(sensor_data):
    """Prepare sensor data for prediction by adding engineered features"""
    # Convert to DataFrame if it's a dictionary
    if isinstance(sensor_data, dict):
        sensor_df = pd.DataFrame([sensor_data])
    else:
        sensor_df = pd.DataFrame(sensor_data)

    # Add engineered features
    sensor_df['NPK_ratio'] = (sensor_df['N'] + sensor_df['P'] + sensor_df['K']) / 3
    sensor_df['N_P_ratio'] = sensor_df['N'] / sensor_df['P']
    sensor_df['N_K_ratio'] = sensor_df['N'] / sensor_df['K']
    sensor_df['P_K_ratio'] = sensor_df['P'] / sensor_df['K']
    sensor_df['moisture_index'] = (sensor_df['rainfall'] / (sensor_df['temperature'] + 0.1)) * 10
    sensor_df['ph_deviation'] = abs(sensor_df['ph'] - 6.5)
    sensor_df['rainfall_adequacy'] = sensor_df['rainfall'].apply(rainfall_adequacy)

    # Calculate NPK balance score for maize
    # Ideal NPK ratio for maize is approximately 1.5:0.5:1
    ideal_n = 1.5
    ideal_p = 0.5
    ideal_k = 1.0
    total_ideal = ideal_n + ideal_p + ideal_k

    # Calculate ideal portions
    ideal_n_portion = ideal_n / total_ideal
    ideal_p_portion = ideal_p / total_ideal
    ideal_k_portion = ideal_k / total_ideal

    sensor_df['total_npk'] = sensor_df['N'] + sensor_df['P'] + sensor_df['K']
    sensor_df['n_portion'] = sensor_df['N'] / sensor_df['total_npk']
    sensor_df['p_portion'] = sensor_df['P'] / sensor_df['total_npk']
    sensor_df['k_portion'] = sensor_df['K'] / sensor_df['total_npk']
    sensor_df['npk_balance_score'] = (
            abs(sensor_df['n_portion'] - ideal_n_portion) +
            abs(sensor_df['p_portion'] - ideal_p_portion) +
            abs(sensor_df['k_portion'] - ideal_k_portion)
    )
    sensor_df['ph_rainfall_interaction'] = sensor_df['ph_deviation'] * (1 - sensor_df['rainfall_adequacy'])

    return sensor_df