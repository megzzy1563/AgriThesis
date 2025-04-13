import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from app.config import FIREBASE_CREDENTIALS_PATH, FERTILIZER_DOC_ID
import logging

logger = logging.getLogger(__name__)


class FirebaseService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
            cls._initialize(cls._instance)
        return cls._instance

    def _initialize(self):
        try:
            # Initialize Firebase if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Firebase: {e}")
            raise

    def update_fertilizer_recommendation(self, recommendation, application_method, quantity_recommendation=None):
        """Update the Firestore document with fertilizer recommendation"""
        try:
            doc_data = {
                "fertilizer_type": recommendation,
                "fertilizer_application": application_method,
                "timestamp": datetime.now()
            }

            # Add quantity recommendation if available
            if quantity_recommendation:
                # Convert the recommendation to a format that Firestore can store
                # (Firestore doesn't support nested objects with custom methods)
                doc_data["quantity_recommendation"] = self._prepare_for_firestore(quantity_recommendation)

            doc_ref = self.db.collection("machine-learning-model").document(FERTILIZER_DOC_ID)
            doc_ref.update(doc_data)

            logger.info(f"Successfully updated fertilizer data in document {FERTILIZER_DOC_ID}")
            return {
                "document_id": FERTILIZER_DOC_ID,
                "success": True,
                "message": "Fertilizer recommendation updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating fertilizer data: {e}")
            return {
                "document_id": FERTILIZER_DOC_ID,
                "success": False,
                "message": f"Error updating fertilizer recommendation: {str(e)}"
            }

    def get_latest_recommendation(self):
        """Get the fertilizer recommendation from the specific document"""
        try:
            doc_ref = self.db.collection("machine-learning-model").document(FERTILIZER_DOC_ID)
            doc = doc_ref.get()

            if doc.exists:
                rec = doc.to_dict()
                rec["id"] = doc.id
                return rec
            else:
                logger.warning(f"Document {FERTILIZER_DOC_ID} does not exist")
                return None
        except Exception as e:
            logger.error(f"Error getting recommendation: {e}")
            return None

    def _prepare_for_firestore(self, data):
        """Convert Pydantic models or complex objects to dictionaries for Firestore"""
        if hasattr(data, 'dict'):
            # It's a Pydantic model, use its dict() method
            return self._prepare_for_firestore(data.dict())
        elif isinstance(data, dict):
            # Process each item in the dictionary
            return {k: self._prepare_for_firestore(v) for k, v in data.items()}
        elif isinstance(data, list):
            # Process each item in the list
            return [self._prepare_for_firestore(item) for item in data]
        else:
            # Return as is for primitive types
            return data