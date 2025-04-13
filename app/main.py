import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os

from app.models.fertilizer import SensorData, PredictionResponse, FirestoreUpdateResponse
from app.services.prediction_service import PredictionService
from app.services.firebase_service import FirebaseService
from app.models.ml_models import MaizeFertilizerModel

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Maize Fertilizer Recommendation API",
    description="API for predicting fertilizer recommendations for maize cultivation",
    version="1.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injections
def get_prediction_service():
    return PredictionService()


def get_firebase_service():
    return FirebaseService()


def get_ml_model_service():
    return MaizeFertilizerModel()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_fertilizer(
        sensor_data: SensorData,
        prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict fertilizer recommendation based on soil sensor data.

    Returns:
    - Fertilizer type recommendation
    - Application method
    - Soil status (pH, rainfall, NPK levels)
    - Quantitative recommendation with specific amounts and application schedule
    """
    try:
        result = prediction_service.predict_and_update_firestore(sensor_data.dict())
        return result
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendation", response_model=PredictionResponse)
async def get_recommendation(
        firebase_service: FirebaseService = Depends(get_firebase_service)
):
    """
    Get the latest fertilizer recommendation from Firestore
    """
    try:
        # Add debugging here
        logger.info("Attempting to get latest recommendation from Firestore")
        result = firebase_service.get_latest_recommendation()
        if result:
            logger.info("Successfully retrieved recommendation")
            return result
        else:
            logger.error("No recommendation found")
            raise HTTPException(status_code=404, detail="No recommendation found")
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(
        background_tasks: BackgroundTasks,
        ml_model_service: MaizeFertilizerModel = Depends(get_ml_model_service)
):
    """
    Train the fertilizer recommendation model (runs in background)
    """
    try:
        # Check if dataset exists
        dataset_path = os.path.join("data", "Crop_recommendation.csv")
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Training dataset not found")

        # Schedule training in background to not block the API
        background_tasks.add_task(ml_model_service.train_model, dataset_path)

        return {
            "message": "Model training started in background",
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


@app.get("/quantity-calculator", response_model=dict)
async def calculate_fertilizer_quantity(
        n: float = 0,
        p: float = 0,
        k: float = 0,
        ph: float = 6.5,
        rainfall: float = 800,
        fertilizer_type: str = "Complete Fertilizer"
):
    """
    Standalone endpoint to calculate fertilizer quantities based on input parameters

    Args:
        n: Nitrogen content in soil (mg/kg or ppm)
        p: Phosphorus content in soil (mg/kg or ppm)
        k: Potassium content in soil (mg/kg or ppm)
        ph: Soil pH
        rainfall: Annual rainfall in mm
        fertilizer_type: Type of fertilizer to use

    Returns:
        Detailed fertilizer quantity recommendation
    """
    try:
        from app.utils.fertilizer_quantity import calculate_fertilizer_recommendation

        # Create sensor data dictionary
        sensor_data = {
            "N": n,
            "P": p,
            "K": k,
            "ph": ph,
            "rainfall": rainfall,
            "temperature": 25.0,  # Default value
            "humidity": 65.0  # Default value
        }

        # Calculate recommendation
        recommendation = calculate_fertilizer_recommendation(sensor_data, fertilizer_type)

        return recommendation
    except Exception as e:
        logger.error(f"Error calculating fertilizer quantity: {e}")
        raise HTTPException(status_code=500, detail=str(e))