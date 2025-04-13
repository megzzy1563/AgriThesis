import joblib
import os
import logging
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH

logger = logging.getLogger(__name__)


class MaizeFertilizerModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaizeFertilizerModel, cls).__new__(cls)
            cls._initialize(cls._instance)
        return cls._instance

    def _initialize(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
            'NPK_ratio', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
            'moisture_index', 'ph_deviation', 'rainfall_adequacy',
            'npk_balance_score', 'ph_rainfall_interaction'
        ]
        self.load_model()

    def load_model(self):
        """Load the trained model, scaler, and label encoder"""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                logger.info("Model, scaler, and label encoder loaded successfully")
                return True
            else:
                logger.warning("Model files not found, model needs to be trained")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def train_model(self, crop_data_path):
        """Train the maize fertilizer recommendation model"""
        from app.utils.data_processing import categorize_npk_maize, categorize_ph_maize, categorize_rainfall_maize, \
            rainfall_adequacy

        try:
            # Load the crop recommendation dataset
            crop_data = pd.read_csv(crop_data_path)
            maize_data = crop_data

            # Add NPK categories
            maize_data['N_category'] = maize_data['N'].apply(lambda x: categorize_npk_maize(x, 'N'))
            maize_data['P_category'] = maize_data['P'].apply(lambda x: categorize_npk_maize(x, 'P'))
            maize_data['K_category'] = maize_data['K'].apply(lambda x: categorize_npk_maize(x, 'K'))
            maize_data['pH_category'] = maize_data['ph'].apply(categorize_ph_maize)
            maize_data['rainfall_category'] = maize_data['rainfall'].apply(categorize_rainfall_maize)

            # Add fertilizer recommendation based on NPK categories, pH, and rainfall
            def recommend_fertilizer_maize(row):
                n_cat = row['N_category']
                p_cat = row['P_category']
                k_cat = row['K_category']
                ph_cat = row['pH_category']
                rain_cat = row['rainfall_category']

                # Base recommendation on NPK status
                if n_cat == 'Low' and p_cat == 'Low' and k_cat == 'Low':
                    base_rec = 'NPK-rich Complete Fertilizer'
                elif n_cat == 'Low':  # Prioritize nitrogen for maize as it's a heavy N feeder
                    if p_cat == 'Low':
                        base_rec = 'NP Fertilizer Mix'
                    elif k_cat == 'Low':
                        base_rec = 'NK Fertilizer Mix'
                    else:
                        base_rec = 'Nitrogen-rich Fertilizer'
                elif p_cat == 'Low' and k_cat == 'Low':
                    base_rec = 'PK Fertilizer Mix'
                elif p_cat == 'Low':
                    base_rec = 'Phosphorus-rich Fertilizer'
                elif k_cat == 'Low':
                    base_rec = 'Potassium-rich Fertilizer'
                else:
                    base_rec = 'Balanced Maintenance Fertilizer'

                # Modify recommendation based on pH
                ph_modifier = ""
                if ph_cat == 'Very Acidic':
                    ph_modifier = " with Lime"
                elif ph_cat == 'Acidic':
                    ph_modifier = " with Calcium Carbonate"
                elif ph_cat == 'Alkaline':
                    ph_modifier = " with Sulfur"

                # Adjust based on rainfall
                rain_modifier = ""
                if rain_cat == 'Insufficient':
                    rain_modifier = " (Split Application Recommended)"
                elif rain_cat == 'Excessive':
                    rain_modifier = " (Slow-Release Formulation)"

                return base_rec + ph_modifier + rain_modifier

            maize_data['fertilizer_recommendation'] = maize_data.apply(recommend_fertilizer_maize, axis=1)

            # Feature engineering
            logger.info("Performing feature engineering for maize-specific model...")

            # Create NPK ratio
            maize_data['NPK_ratio'] = (maize_data['N'] + maize_data['P'] + maize_data['K']) / 3
            maize_data['N_P_ratio'] = maize_data['N'] / maize_data['P']
            maize_data['N_K_ratio'] = maize_data['N'] / maize_data['K']
            maize_data['P_K_ratio'] = maize_data['P'] / maize_data['K']

            # Create soil moisture indicator
            maize_data['moisture_index'] = (maize_data['rainfall'] / (maize_data['temperature'] + 0.1)) * 10

            # Add pH deviation
            maize_data['ph_deviation'] = abs(maize_data['ph'] - 6.5)

            # Add rainfall adequacy
            maize_data['rainfall_adequacy'] = maize_data['rainfall'].apply(rainfall_adequacy)

            # Add NPK balance score
            # Ideal NPK ratio for maize is approximately 1.5:0.5:1
            ideal_n = 1.5
            ideal_p = 0.5
            ideal_k = 1.0
            total_ideal = ideal_n + ideal_p + ideal_k

            # Calculate ideal portions
            ideal_n_portion = ideal_n / total_ideal
            ideal_p_portion = ideal_p / total_ideal
            ideal_k_portion = ideal_k / total_ideal

            # Calculate actual portions
            maize_data['total_npk'] = maize_data['N'] + maize_data['P'] + maize_data['K']
            maize_data['n_portion'] = maize_data['N'] / maize_data['total_npk']
            maize_data['p_portion'] = maize_data['P'] / maize_data['total_npk']
            maize_data['k_portion'] = maize_data['K'] / maize_data['total_npk']

            # Calculate deviation from ideal
            maize_data['npk_balance_score'] = (
                    abs(maize_data['n_portion'] - ideal_n_portion) +
                    abs(maize_data['p_portion'] - ideal_p_portion) +
                    abs(maize_data['k_portion'] - ideal_k_portion)
            )

            # Add pH and rainfall interaction
            maize_data['ph_rainfall_interaction'] = maize_data['ph_deviation'] * (1 - maize_data['rainfall_adequacy'])

            # Prepare data for modeling
            X = maize_data[self.feature_columns]
            y = maize_data['fertilizer_recommendation']

            # Encode the target variable
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42,
                                                                stratify=y_encoded)

            # Feature scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            logger.info("Training AdaBoost model optimized for maize...")
            base_estimator = DecisionTreeClassifier(max_depth=3)
            self.model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=100,
                learning_rate=0.8,
                random_state=42
            )

            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)

            # Model evaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(
                f"Model Performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Save the trained model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
            logger.info("Model, scaler, and label encoder saved to disk")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, processed_data):
        """Make prediction with the model"""
        try:
            # Scale the data
            scaled_data = self.scaler.transform(processed_data[self.feature_columns])

            # Make prediction
            prediction = self.model.predict(scaled_data)
            prediction_label = self.label_encoder.inverse_transform(prediction)[0]

            return prediction_label
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise