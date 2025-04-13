from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Any, List


class SensorData(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil", ge=0)
    P: float = Field(..., description="Phosphorus content in soil", ge=0)
    K: float = Field(..., description="Potassium content in soil", ge=0)
    temperature: float = Field(..., description="Temperature in Celsius", ge=0)
    humidity: float = Field(..., description="Humidity percentage", ge=0, le=100)
    ph: float = Field(..., description="pH level of soil", ge=0, le=14)
    rainfall: float = Field(..., description="Rainfall in mm", ge=0)

    @validator('ph')
    def validate_ph(cls, v):
        if v < 0 or v > 14:
            raise ValueError("pH must be between 0 and 14")
        return v


class NPKStatus(BaseModel):
    N: str
    P: str
    K: str


class FertilizerSchedule(BaseModel):
    timing: str
    percentage: Optional[str] = None
    quantity: float
    fertilizer: str
    secondary: Optional[Dict[str, Any]] = None


class FertilizerQuantity(BaseModel):
    name: str
    quantity: float
    unit: str


class FertilizerRecommendation(BaseModel):
    primary_fertilizer: FertilizerQuantity
    secondary_fertilizer: Optional[FertilizerQuantity] = None
    application_schedule: Dict[str, Any]
    soil_amendment: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    fertilizer_type: str
    fertilizer_application: str
    pH_status: str
    rainfall_status: str
    npk_status: NPKStatus
    quantity_recommendation: Optional[FertilizerRecommendation] = None
    id: Optional[str] = None


class FirestoreUpdateResponse(BaseModel):
    document_id: str
    success: bool
    message: str