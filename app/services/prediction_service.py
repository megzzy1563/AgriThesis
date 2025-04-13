import logging
from app.models.ml_models import MaizeFertilizerModel
from app.services.firebase_service import FirebaseService
from app.utils.data_processing import (
    prepare_sensor_data_for_prediction,
    categorize_ph_maize,
    categorize_rainfall_maize,
    categorize_npk_maize,
    get_application_method
)
from app.utils.fertilizer_quantity import calculate_fertilizer_recommendation

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.model_service = MaizeFertilizerModel()
        self.firebase_service = FirebaseService()

    def predict_and_update_firestore(self, sensor_data):
        """Make prediction with the model and update in Firestore"""
        try:
            # Check if model is loaded, if not try to load it
            if not self.model_service.model:
                if not self.model_service.load_model():
                    raise Exception("Model not found and could not be loaded")

            # Prepare data
            processed_data = prepare_sensor_data_for_prediction(sensor_data)

            # Make prediction
            prediction_label = self.model_service.predict(processed_data)
            logger.info(f"Prediction successful: {prediction_label}")

            # Get status information
            pH_status = categorize_ph_maize(sensor_data['ph'])
            rainfall_status = categorize_rainfall_maize(sensor_data['rainfall'])

            # Get application method
            application_method = get_application_method(rainfall_status, pH_status)

            # Calculate fertilizer quantity recommendation
            quantity_recommendation = calculate_fertilizer_recommendation(
                sensor_data,
                prediction_label
            )
            logger.info("Quantity recommendation calculated successfully")

            # Update Firestore
            logger.info("Attempting to update Firestore...")
            firestore_response = self.firebase_service.update_fertilizer_recommendation(
                prediction_label,
                application_method,
                quantity_recommendation
            )
            logger.info(f"Firestore update response: {firestore_response}")

            # Create response
            response = {
                "fertilizer_type": prediction_label,
                "fertilizer_application": application_method,
                "pH_status": pH_status,
                "rainfall_status": rainfall_status,
                "npk_status": {
                    "N": categorize_npk_maize(sensor_data['N'], 'N'),
                    "P": categorize_npk_maize(sensor_data['P'], 'P'),
                    "K": categorize_npk_maize(sensor_data['K'], 'K')
                },
                "quantity_recommendation": quantity_recommendation,
                "id": firestore_response["document_id"] if firestore_response["success"] else None
            }

            return response
        except Exception as e:
            logger.error(f"Error in prediction service: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise