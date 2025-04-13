"""
Fertilizer quantity recommendation module based on Philippine standards
for maize/corn cultivation. Recommendations follow Department of Agriculture
and Philippine Rice Research Institute guidelines.
"""

import logging
from typing import Dict, Any, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Philippine standard target values for maize (in kg/ha)
# Based on Philippines Department of Agriculture guidelines
PH_MAIZE_TARGETS = {
    "N": 120,  # kg/ha for high-yielding varieties
    "P": 60,   # kg/ha (expressed as P2O5)
    "K": 60    # kg/ha (expressed as K2O)
}

# Conversion factors for elemental to oxide forms
# These are standard agricultural chemistry conversion factors
P_TO_P2O5 = 2.29
K_TO_K2O = 1.20

# NPK thresholds for maize - these should match the thresholds in data_processing.py
NPK_THRESHOLDS = {
    "N": {"low": 80, "medium": 140},
    "P": {"low": 15, "medium": 30},
    "K": {"low": 80, "medium": 150}
}

# Standard fertilizer compositions (N-P2O5-K2O percentages)
FERTILIZER_COMPOSITIONS = {
    "Urea": {"N": 46, "P": 0, "K": 0},
    "Ammonium Sulfate": {"N": 21, "P": 0, "K": 0},
    "Ammonium Phosphate (16-20-0)": {"N": 16, "P": 20, "K": 0},
    "Complete Fertilizer (14-14-14)": {"N": 14, "P": 14, "K": 14},
    "Complete Fertilizer (16-16-16)": {"N": 16, "P": 16, "K": 16},
    "Muriate of Potash (0-0-60)": {"N": 0, "P": 0, "K": 60},
    "Single Superphosphate": {"N": 0, "P": 18, "K": 0},
    "Triple Superphosphate": {"N": 0, "P": 46, "K": 0},
}

# Application timing recommendations based on crop growth stage
APPLICATION_TIMING = {
    "Basal Application": "Apply at planting time or before planting",
    "First Top Dressing": "Apply 25-30 days after planting",
    "Second Top Dressing": "Apply 45-50 days after planting (before tasseling)"
}

def calculate_npk_deficit_from_category(npk_values: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate nutrient deficit based on the categorical level (Low, Medium, High)
    rather than direct conversion of sensor values.

    Args:
        npk_values: Dictionary with N, P, K values in mg/kg (ppm)

    Returns:
        Dictionary of N, P, K deficits in kg/ha
    """
    # Initialize deficits dictionary
    deficit = {"N": 0, "P": 0, "K": 0}

    # Calculate deficit for each nutrient based on thresholds
    for nutrient, value in npk_values.items():
        if nutrient not in ['N', 'P', 'K']:
            continue

        thresholds = NPK_THRESHOLDS.get(nutrient, {})
        target = PH_MAIZE_TARGETS.get(nutrient, 0)

        # If value is below low threshold, apply full deficit
        if value < thresholds.get("low", 0):
            deficit[nutrient] = target
        # If value is between low and medium, apply partial deficit
        elif value < thresholds.get("medium", 0):
            deficit[nutrient] = target * 0.5
        # Otherwise, small maintenance amount
        else:
            deficit[nutrient] = target * 0.1

    # Convert P and K to oxide form
    deficit['P2O5'] = deficit['P'] * P_TO_P2O5
    deficit['K2O'] = deficit['K'] * K_TO_K2O

    return deficit

def adjust_for_soil_conditions(deficit: Dict[str, float],
                              ph: float,
                              rainfall: float) -> Dict[str, float]:
    """
    Adjust nutrient recommendations based on soil pH and rainfall

    Args:
        deficit: Dictionary with nutrient deficits
        ph: Soil pH
        rainfall: Annual rainfall in mm

    Returns:
        Adjusted deficit dictionary
    """
    adjusted = deficit.copy()

    # pH adjustments (phosphorus availability is highly pH dependent)
    if ph < 5.5:
        # Very acidic soils - increase P due to fixation
        adjusted['P2O5'] = adjusted.get('P2O5', 0) * 1.3
        # Liming will be recommended separately

    elif ph > 7.5:
        # Alkaline soils - increase P due to calcium fixation
        adjusted['P2O5'] = adjusted.get('P2O5', 0) * 1.25

    # Rainfall adjustments
    if rainfall > 1200:
        # High rainfall areas - increase N due to leaching
        adjusted['N'] = adjusted.get('N', 0) * 1.2

    elif rainfall < 500:
        # Low rainfall - adjust for moisture-limited conditions
        # Recommend split application rather than increasing rates
        pass

    return adjusted

def get_fertilizer_quantities(adjusted_deficit: Dict[str, float],
                             fertilizer_type: str) -> Dict[str, Any]:
    """
    Calculate fertilizer quantities based on adjusted deficits

    Args:
        adjusted_deficit: Dictionary with adjusted nutrient deficits
        fertilizer_type: Recommended fertilizer type

    Returns:
        Dictionary with fertilizer recommendations
    """
    # Extract needed values with defaults for safety
    n_deficit = adjusted_deficit.get('N', 0)
    p_deficit = adjusted_deficit.get('P2O5', 0)
    k_deficit = adjusted_deficit.get('K2O', 0)

    # Initialize return dictionary
    recommendation = {
        "primary_fertilizer": {"name": "", "quantity": 0, "unit": "kg/ha"},
        "secondary_fertilizer": {"name": "", "quantity": 0, "unit": "kg/ha"},
        "application_schedule": {}
    }

    # Determine fertilizers based on the recommendation type
    if "NPK-rich" in fertilizer_type or "Complete Fertilizer" in fertilizer_type:
        # Use complete fertilizer as primary
        if n_deficit >= p_deficit and n_deficit >= k_deficit:
            # Higher N need - use 16-16-16
            primary = "Complete Fertilizer (16-16-16)"
            comp = FERTILIZER_COMPOSITIONS[primary]
            # Calculate based on P requirement (usually limiting)
            quantity = (p_deficit / comp["P"]) * 100

            # Secondary N fertilizer if needed
            remaining_n = n_deficit - (quantity * comp["N"] / 100)
            if remaining_n > 10:  # Threshold for adding more N
                secondary = "Urea"
                sec_quantity = (remaining_n / FERTILIZER_COMPOSITIONS[secondary]["N"]) * 100
                recommendation["secondary_fertilizer"]["name"] = secondary
                recommendation["secondary_fertilizer"]["quantity"] = round(sec_quantity, 1)
        else:
            # Balanced need - use 14-14-14
            primary = "Complete Fertilizer (14-14-14)"
            comp = FERTILIZER_COMPOSITIONS[primary]
            # Calculate based on highest deficit
            divisor = max(comp["N"] / n_deficit if n_deficit else 0,
                         comp["P"] / p_deficit if p_deficit else 0,
                         comp["K"] / k_deficit if k_deficit else 0)
            quantity = (1 / divisor) * 100 if divisor else 0

    elif "Nitrogen-rich" in fertilizer_type:
        # Use Urea as primary
        primary = "Urea"
        quantity = (n_deficit / FERTILIZER_COMPOSITIONS[primary]["N"]) * 100

        # Check if P is also needed
        if p_deficit > 10:
            secondary = "Single Superphosphate"
            sec_quantity = (p_deficit / FERTILIZER_COMPOSITIONS[secondary]["P"]) * 100
            recommendation["secondary_fertilizer"]["name"] = secondary
            recommendation["secondary_fertilizer"]["quantity"] = round(sec_quantity, 1)

    elif "Phosphorus-rich" in fertilizer_type:
        # Use Triple Superphosphate as primary
        primary = "Triple Superphosphate"
        quantity = (p_deficit / FERTILIZER_COMPOSITIONS[primary]["P"]) * 100

        # Check if N is also needed
        if n_deficit > 10:
            secondary = "Urea"
            sec_quantity = (n_deficit / FERTILIZER_COMPOSITIONS[secondary]["N"]) * 100
            recommendation["secondary_fertilizer"]["name"] = secondary
            recommendation["secondary_fertilizer"]["quantity"] = round(sec_quantity, 1)

    elif "Potassium-rich" in fertilizer_type:
        # Use Muriate of Potash as primary
        primary = "Muriate of Potash (0-0-60)"
        quantity = (k_deficit / FERTILIZER_COMPOSITIONS[primary]["K"]) * 100

        # Check if N is also needed
        if n_deficit > 10:
            secondary = "Urea"
            sec_quantity = (n_deficit / FERTILIZER_COMPOSITIONS[secondary]["N"]) * 100
            recommendation["secondary_fertilizer"]["name"] = secondary
            recommendation["secondary_fertilizer"]["quantity"] = round(sec_quantity, 1)

    elif "NP Fertilizer" in fertilizer_type:
        # Use Ammonium Phosphate
        primary = "Ammonium Phosphate (16-20-0)"
        comp = FERTILIZER_COMPOSITIONS[primary]
        # Calculate based on P requirement
        quantity = (p_deficit / comp["P"]) * 100

        # Additional N if needed
        remaining_n = n_deficit - (quantity * comp["N"] / 100)
        if remaining_n > 10:
            secondary = "Urea"
            sec_quantity = (remaining_n / FERTILIZER_COMPOSITIONS[secondary]["N"]) * 100
            recommendation["secondary_fertilizer"]["name"] = secondary
            recommendation["secondary_fertilizer"]["quantity"] = round(sec_quantity, 1)

    else:
        # Default to complete fertilizer if no specific match
        primary = "Complete Fertilizer (14-14-14)"
        # Base on average needs
        quantity = ((n_deficit/14) + (p_deficit/14) + (k_deficit/14)) * 100 / 3

    # Set primary fertilizer
    recommendation["primary_fertilizer"]["name"] = primary
    recommendation["primary_fertilizer"]["quantity"] = round(quantity, 1)

    # Create application schedule (based on Philippine practices for maize)
    if "Split Application" in fertilizer_type or quantity > 200:
        # Create split application schedule
        recommendation["application_schedule"] = {
            "basal": {
                "timing": APPLICATION_TIMING["Basal Application"],
                "percentage": "40%",
                "quantity": round(quantity * 0.4, 1),
                "fertilizer": primary
            },
            "first_top_dressing": {
                "timing": APPLICATION_TIMING["First Top Dressing"],
                "percentage": "30%",
                "quantity": round(quantity * 0.3, 1),
                "fertilizer": primary
            },
            "second_top_dressing": {
                "timing": APPLICATION_TIMING["Second Top Dressing"],
                "percentage": "30%",
                "quantity": round(quantity * 0.3, 1),
                "fertilizer": primary
            }
        }

        # Also split secondary fertilizer if applicable
        if recommendation["secondary_fertilizer"]["name"]:
            sec_qty = recommendation["secondary_fertilizer"]["quantity"]
            sec_name = recommendation["secondary_fertilizer"]["name"]

            # For nitrogen sources, apply more in later stages
            if "Urea" in sec_name or "Ammonium" in sec_name:
                recommendation["application_schedule"]["first_top_dressing"]["secondary"] = {
                    "quantity": round(sec_qty * 0.5, 1),
                    "fertilizer": sec_name
                }
                recommendation["application_schedule"]["second_top_dressing"]["secondary"] = {
                    "quantity": round(sec_qty * 0.5, 1),
                    "fertilizer": sec_name
                }
            else:
                # For P and K, apply more at basal
                recommendation["application_schedule"]["basal"]["secondary"] = {
                    "quantity": round(sec_qty * 0.7, 1),
                    "fertilizer": sec_name
                }
                recommendation["application_schedule"]["first_top_dressing"]["secondary"] = {
                    "quantity": round(sec_qty * 0.3, 1),
                    "fertilizer": sec_name
                }
    else:
        # Standard application (not split)
        recommendation["application_schedule"] = {
            "basal": {
                "timing": APPLICATION_TIMING["Basal Application"],
                "percentage": "70%",
                "quantity": round(quantity * 0.7, 1),
                "fertilizer": primary
            },
            "top_dressing": {
                "timing": APPLICATION_TIMING["First Top Dressing"],
                "percentage": "30%",
                "quantity": round(quantity * 0.3, 1),
                "fertilizer": primary
            }
        }

        # Add secondary fertilizer if applicable
        if recommendation["secondary_fertilizer"]["name"]:
            sec_qty = recommendation["secondary_fertilizer"]["quantity"]
            sec_name = recommendation["secondary_fertilizer"]["name"]

            # For P and K, apply more at basal
            if "Urea" not in sec_name and "Ammonium" not in sec_name:
                recommendation["application_schedule"]["basal"]["secondary"] = {
                    "quantity": round(sec_qty, 1),
                    "fertilizer": sec_name
                }
            else:
                # For N, apply at top dressing
                recommendation["application_schedule"]["top_dressing"]["secondary"] = {
                    "quantity": round(sec_qty, 1),
                    "fertilizer": sec_name
                }

    return recommendation

def calculate_fertilizer_recommendation(
    sensor_data: Dict[str, float],
    fertilizer_type: str
) -> Dict[str, Any]:
    """
    Master function that calculates fertilizer quantities based on sensor data

    Args:
        sensor_data: Dictionary containing N, P, K, pH, rainfall etc.
        fertilizer_type: The type of fertilizer recommended by the ML model

    Returns:
        Dictionary with complete fertilizer recommendation
    """
    try:
        # 1. Calculate nutrient deficit based on categories
        # This is the main fix - using categorical deficits instead of direct conversion
        npk_values = {"N": sensor_data["N"], "P": sensor_data["P"], "K": sensor_data["K"]}
        deficit = calculate_npk_deficit_from_category(npk_values)

        # 2. Adjust for soil conditions
        adjusted = adjust_for_soil_conditions(
            deficit,
            sensor_data["ph"],
            sensor_data["rainfall"]
        )

        # 3. Get fertilizer quantities
        recommendations = get_fertilizer_quantities(adjusted, fertilizer_type)

        # 4. Add soil amendment recommendations based on pH
        if sensor_data["ph"] < 5.5:
            # Calculate lime requirement (approximate formula for Philippine soils)
            # Lime in tons/ha = (6.5 - soil pH) * 1.5
            lime_tons = round((6.5 - sensor_data["ph"]) * 1.5, 1)
            if lime_tons > 0:
                recommendations["soil_amendment"] = {
                    "name": "Agricultural Lime",
                    "quantity": lime_tons,
                    "unit": "tons/ha",
                    "application": "Apply and incorporate into soil 2-4 weeks before planting"
                }
        elif sensor_data["ph"] > 7.5:
            # Sulfur recommendation for alkaline soils
            # Approximate value for Philippine soils
            sulfur_kg = round((sensor_data["ph"] - 6.5) * 300, 1)
            if sulfur_kg > 0:
                recommendations["soil_amendment"] = {
                    "name": "Agricultural Sulfur",
                    "quantity": sulfur_kg,
                    "unit": "kg/ha",
                    "application": "Apply and incorporate into soil 4-6 weeks before planting"
                }

        return recommendations

    except Exception as e:
        logger.error(f"Error calculating fertilizer recommendation: {e}")
        # Return a simplified recommendation if calculation fails
        return {
            "primary_fertilizer": {
                "name": "Complete Fertilizer (14-14-14)",
                "quantity": 250,
                "unit": "kg/ha"
            },
            "application_schedule": {
                "basal": {
                    "timing": "Apply at planting time",
                    "percentage": "60%",
                    "quantity": 150,
                    "fertilizer": "Complete Fertilizer (14-14-14)"
                },
                "top_dressing": {
                    "timing": "Apply 30 days after planting",
                    "percentage": "40%",
                    "quantity": 100,
                    "fertilizer": "Complete Fertilizer (14-14-14)"
                }
            }
        }