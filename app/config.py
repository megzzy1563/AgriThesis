import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Firebase settings
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "adaboost_maize_fertilizer_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "maize_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "maize_label_encoder.pkl")

# Firestore document ID
FERTILIZER_DOC_ID = os.getenv("FERTILIZER_DOC_ID", "VM5JaRgm4URI58Gg1HcV")