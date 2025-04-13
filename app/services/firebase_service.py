import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json
import os
from app.config import FIREBASE_CREDENTIALS_PATH, FERTILIZER_DOC_ID
import logging
import traceback
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class FirebaseService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating new FirebaseService instance")
            instance = super(FirebaseService, cls).__new__(cls)
            try:
                # Initialize Firebase if not already initialized
                if not firebase_admin._apps:
                    logger.info("Initializing Firebase app")
                    # First try to read from environment variable
                    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                    if cred_json:
                        logger.info("Using credentials from environment variable")
                        try:
                            # Parse JSON string from environment variable
                            cred_dict = json.loads(cred_json)
                            cred = credentials.Certificate(cred_dict)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse credentials JSON: {e}")
                            logger.info("Falling back to credentials file")
                            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
                    else:
                        logger.info("Using credentials from file path")
                        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)

                    firebase_admin.initialize_app(cred)

                instance.db = firestore.client()
                logger.info("Firebase initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Firebase: {e}")
                logger.error(traceback.format_exc())
                # Create a dummy db attribute to prevent attribute errors
                instance.db = None
                logger.warning("Setting db to None to prevent attribute errors")

            cls._instance = instance

        return cls._instance

    def update_fertilizer_recommendation(self, recommendation, application_method, quantity_recommendation=None):
        """Update the Firestore document with fertilizer recommendation"""
        try:
            # Check if db is initialized
            if self.db is None:
                logger.error("Firestore client is not initialized")
                return {
                    "document_id": FERTILIZER_DOC_ID,
                    "success": False,
                    "message": "Firestore client is not initialized"
                }

            doc_data = {
                "fertilizer_type": recommendation,
                "fertilizer_application": application_method,
                "timestamp": datetime.now()
            }

            # Add quantity recommendation if available
            if quantity_recommendation:
                # Convert the recommendation to a format that Firestore can store
                doc_data["quantity_recommendation"] = self._prepare_for_firestore(quantity_recommendation)

            doc_ref = self.db.collection("machine-learning-model").document(FERTILIZER_DOC_ID)

            # Check if document exists
            doc = doc_ref.get()
            if doc.exists:
                # Update existing document
                doc_ref.update(doc_data)
                logger.info(f"Updated existing document {FERTILIZER_DOC_ID}")
            else:
                # Create new document
                doc_ref.set(doc_data)
                logger.info(f"Created new document {FERTILIZER_DOC_ID}")

            logger.info(f"Successfully updated fertilizer data in document {FERTILIZER_DOC_ID}")
            return {
                "document_id": FERTILIZER_DOC_ID,
                "success": True,
                "message": "Fertilizer recommendation updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating fertilizer data: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "document_id": FERTILIZER_DOC_ID,
                "success": False,
                "message": f"Error updating fertilizer recommendation: {str(e)}"
            }

    def get_latest_recommendation(self):
        """Get the fertilizer recommendation from the specific document"""
        try:
            # Check if db is initialized
            if self.db is None:
                logger.error("Firestore client is not initialized")
                raise Exception("Firestore client is not initialized")

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
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=404, detail="No recommendation found")

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