import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
import uuid
import os
import json
import logging
import joblib # For loading/saving the ML model
import subprocess # Needed for launching Streamlit from python command
import sys # Needed for sys.exit and sys.executable

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration and Constants ---
APP_DATA_FILE = "app_data.json"
MODEL_PATH = "models/random_forest_model.joblib"
# Ensure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Define an environment variable to signal Streamlit app mode
STREAMLIT_APP_MODE_ENV = "AGRIPREDICT_STREAMLIT_APP_MODE"

# --- JSON Data Persistence Functions ---
def load_app_data():
    """Loads all application data from the JSON file."""
    if os.path.exists(APP_DATA_FILE):
        try:
            with open(APP_DATA_FILE, 'r') as f:
                data = json.load(f)
                # Convert timestamps back to datetime objects for consistency
                for key in ['farms', 'predictions', 'actual_yields', 'shipments']:
                    if key in data:
                        for record in data[key]:
                            if 'timestamp' in record and isinstance(record['timestamp'], str):
                                record['timestamp'] = datetime.datetime.fromisoformat(record['timestamp'])
                            if 'planting_date' in record and isinstance(record['planting_date'], str):
                                 record['planting_date'] = datetime.datetime.fromisoformat(record['planting_date'])
                logging.info(f"Data loaded from {APP_DATA_FILE}")
                return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {APP_DATA_FILE}: {e}. File might be corrupted. Starting with empty data.")
            # If corrupted, return empty structure to allow app to continue
            return {
                "farms": [],
                "predictions": [],
                "actual_yields": [],
                "shipments": []
            }
    else:
        logging.warning(f"No {APP_DATA_FILE} found. Initializing empty data structure.")
        return {
            "farms": [],
            "predictions": [],
            "actual_yields": [],
            "shipments": []
        }

def save_app_data(data):
    """Saves all application data to the JSON file."""
    
    # Custom JSON serializer to handle datetime objects, UUID objects,
    # and numpy scalar types (like np.bool_, np.int_, np.float_)
    def json_serializer(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj) # Convert UUID objects to string
        # Check for numpy scalar types and convert them to native Python types
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item() # .item() converts numpy scalar to a standard Python scalar
        # If the object is not a recognized type, let the default encoder raise TypeError
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(APP_DATA_FILE, 'w') as f:
        # Use the custom json_serializer function for objects not natively serializable
        json.dump(data, f, indent=4, default=json_serializer)
    logging.info(f"Data saved to {APP_DATA_FILE}")

# --- Data Ingestion and Preprocessing Component ---
class DataProcessor:
    def __init__(self, config=None):
        self.config = config if config else {}
        logging.info("DataProcessor initialized.")

    def fetch_weather_data(self, lat, lon, start_date, end_date):
        logging.info(f"Fetching mock weather data for ({lat}, {lon}) from {start_date} to {end_date}")
        # Generate dummy weather data
        dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        weather_data = {
            'date': dates,
            'temperature_avg': np.random.uniform(20, 30, len(dates)),
            'humidity_avg': np.random.uniform(60, 90, len(dates)),
            'precipitation': np.random.uniform(0, 10, len(dates))
        }
        return pd.DataFrame(weather_data)

    def fetch_satellite_data(self, bounding_box, start_date, end_date):
        logging.info(f"Fetching mock satellite data for {bounding_box} from {start_date} to {end_date}")
        # Generate dummy satellite data (e.g., NDVI)
        dates = [start_date + datetime.timedelta(days=i * 7) for i in range(((end_date - start_date).days // 7) + 1)]
        satellite_data = {
            'date': dates,
            'ndvi': np.random.uniform(0.3, 0.8, len(dates))
        }
        return pd.DataFrame(satellite_data)

    def load_farm_practices_data(self, farm_ids, app_data):
        logging.info(f"Loading farm practices data for {farm_ids}")
        # Filter existing farm data or create dummy if not found
        existing_farms = [f for f in app_data['farms'] if f['farm_id'] in farm_ids]
        
        # If no existing data, create a dummy entry for the requested farm_id
        if not existing_farms:
            new_farm_id = farm_ids[0] if farm_ids else f"Farm_{uuid.uuid4().hex[:8]}"
            dummy_farm_data = {
                "farm_id": new_farm_id,
                "crop_type": np.random.choice(['Maize', 'Rice', 'Wheat', 'Beans']),
                "soil_type": np.random.choice(['Loam', 'Clay', 'Sand']),
                "ph": round(np.random.uniform(5.5, 7.5), 1),
                "nitrogen": round(np.random.uniform(50, 150), 2),
                "phosphorus": round(np.random.uniform(30, 80), 2),
                "planting_date": datetime.datetime.now() - datetime.timedelta(days=np.random.randint(60, 120)),
                "fertilizer_applied": bool(np.random.choice([True, False])), # Ensure Python bool
                "irrigation_method": np.random.choice(['Rainfed', 'Drip', 'Sprinkler']),
                "previous_yield": round(np.random.uniform(1.0, 3.0), 2),
                "last_updated": datetime.datetime.now()
            }
            app_data['farms'].append(dummy_farm_data)
            logging.info(f"Created dummy farm data for {new_farm_id}")
            save_app_data(app_data)
            return pd.DataFrame([dummy_farm_data])
        
        return pd.DataFrame(existing_farms)


    def load_soil_data(self, farm_ids, app_data):
        logging.info(f"Loading soil data for {farm_ids}")
        # Soil data is typically static or part of farm practices data for this simulation
        return self.load_farm_practices_data(farm_ids, app_data) # Re-use farm practices for basic soil info

    def preprocess_data(self, weather_df, satellite_df, farm_data_df, current_farm_id, current_date, app_data):
        logging.info(f"Preprocessing data for farm: {current_farm_id}")
        
        # Merge weather and satellite data
        # Ensure 'date' column is datetime and set as index for resampling
        if 'date' in weather_df.columns:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_df = weather_df.set_index('date').resample('D').mean().interpolate()

        if 'date' in satellite_df.columns:
            satellite_df['date'] = pd.to_datetime(satellite_df['date'])
            satellite_df = satellite_df.set_index('date').resample('D').mean().interpolate()

        # Combine environmental data
        environmental_df = weather_df.merge(satellite_df, left_index=True, right_index=True, how='outer')
        environmental_df = environmental_df.resample('D').mean().interpolate(method='linear')
        
        # Calculate derived features from environmental data
        environmental_df['gdd'] = (environmental_df['temperature_avg'] - 10).clip(lower=0) # Growing Degree Days base 10C
        
        # Select the latest environmental features for prediction
        latest_env_data = environmental_df.iloc[-1] if not environmental_df.empty else pd.Series()

        # Extract relevant features from farm_data_df for the current farm
        farm_features = farm_data_df[farm_data_df['farm_id'] == current_farm_id].iloc[0] if not farm_data_df.empty else pd.Series()

        if farm_features.empty:
            logging.error(f"Farm data not found for {current_farm_id} during preprocessing.")
            raise ValueError(f"Farm data not found for {current_farm_id}.")

        # Create a single feature vector for prediction
        features = {
            'temperature_avg': latest_env_data.get('temperature_avg', np.random.uniform(20,30)),
            'humidity_avg': latest_env_data.get('humidity_avg', np.random.uniform(60,90)),
            'precipitation': latest_env_data.get('precipitation', np.random.uniform(0,10)),
            'ndvi': latest_env_data.get('ndvi', np.random.uniform(0.4,0.7)),
            'gdd': latest_env_data.get('gdd', np.random.uniform(10,20)), # Add GDD
            'ph': farm_features.get('ph', 6.5),
            'nitrogen': farm_features.get('nitrogen', 100),
            'phosphorus': farm_features.get('phosphorus', 50),
            'fertilizer_applied': 1 if farm_features.get('fertilizer_applied', False) else 0,
            'previous_yield': farm_features.get('previous_yield', 2.0),
            'days_since_planting': (current_date - farm_features.get('planting_date', current_date)).days
        }

        # Handle categorical features (crop_type, soil_type, irrigation_method) if your model uses them
        # For simplicity, let's just add them as is, assuming the model handles them or they are encoded later
        features['crop_type'] = farm_features.get('crop_type', 'Maize')
        features['soil_type'] = farm_features.get('soil_type', 'Loam')
        features['irrigation_method'] = farm_features.get('irrigation_method', 'Rainfed')

        return pd.DataFrame([features])


# --- ML Model Training and Prediction Component ---
class YieldPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.feature_columns = [
            'temperature_avg', 'humidity_avg', 'precipitation', 'ndvi', 'gdd',
            'ph', 'nitrogen', 'phosphorus', 'fertilizer_applied', 'previous_yield',
            'days_since_planting'
        ]
        # Example categories for one-hot encoding if needed
        self.crop_types = ['Maize', 'Rice', 'Wheat', 'Beans']
        self.soil_types = ['Loam', 'Clay', 'Sand']
        self.irrigation_methods = ['Rainfed', 'Drip', 'Sprinkler']
        self._load_model()
        logging.info("YieldPredictor initialized.")

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logging.info(f"ML model loaded from {self.model_path}")
        else:
            logging.warning(f"ML model not found at {self.model_path}. Model will be trained.")
            self.model = None # Ensure it's None if not found

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logging.info(f"ML model saved to {self.model_path}")

    def train_model(self, data_df=None):
        logging.info("Training ML model...")
        
        if data_df is None or data_df.empty:
            logging.info("Generating synthetic data for model training.")
            num_samples = 100
            data = {
                'temperature_avg': np.random.uniform(20, 30, num_samples),
                'humidity_avg': np.random.uniform(60, 90, num_samples),
                'precipitation': np.random.uniform(0, 10, num_samples),
                'ndvi': np.random.uniform(0.3, 0.8, num_samples),
                'gdd': np.random.uniform(10, 25, num_samples),
                'ph': np.random.uniform(5.5, 7.5, num_samples),
                'nitrogen': np.random.uniform(50, 150, num_samples),
                'phosphorus': np.random.uniform(30, 80, num_samples),
                'fertilizer_applied': np.random.randint(0, 2, num_samples),
                'previous_yield': np.random.uniform(1.0, 3.0, num_samples),
                'days_since_planting': np.random.randint(60, 180, num_samples),
                'crop_type': np.random.choice(self.crop_types, num_samples),
                'soil_type': np.random.choice(self.soil_types, num_samples),
                'irrigation_method': np.random.choice(self.irrigation_methods, num_samples),
                'yield': np.random.uniform(1.5, 4.0, num_samples) # Target variable
            }
            data_df = pd.DataFrame(data)
        
        # One-hot encode categorical features if they are used by the model
        # For simplicity, we'll select only numerical features for this basic RandomForest
        # If you integrate more complex models (e.g., CatBoost, LightGBM, or Scikit-learn with OneHotEncoder),
        # you'd include these categorical columns in your feature set.
        
        X = data_df[self.feature_columns]
        y = data_df['yield']

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self._save_model()

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model training complete. MAE: {mae:.2f}, R2: {r2:.2f}")
        return mae, r2

    def make_prediction(self, input_df, app_data):
        logging.info("Making prediction...")
        if self.model is None:
            logging.error("Model not trained. Training initial model.")
            self.train_model() # Train if not already trained

        # Ensure input_df has all required feature columns in the correct order
        # Handle categorical features if present in input_df by removing them for this simple model
        X_predict = input_df[self.feature_columns]
        
        predicted_yield = self.model.predict(X_predict)
        
        # Save prediction to app_data
        prediction_record = {
            "id": str(uuid.uuid4()),
            "farm_id": input_df['farm_id'].iloc[0], # Assuming farm_id is in input_df
            "predicted_yield": predicted_yield[0],
            "timestamp": datetime.datetime.now()
        }
        app_data['predictions'].append(prediction_record)
        save_app_data(app_data)

        return predicted_yield


# --- Blockchain Interface Component (Mocked) ---
class BlockchainInterface:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.mock_mode = self.config.get('mock_mode', True)
        self.mock_chain = [] # Simple list to simulate blockchain
        logging.info(f"BlockchainInterface initialized (Mock Mode: {self.mock_mode}).")

    def _add_block(self, data):
        """Adds a block to the mock blockchain."""
        block = {
            "index": len(self.mock_chain),
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data,
            "previous_hash": self.mock_chain[-1]['hash'] if self.mock_chain else "0" * 64,
            "hash": str(uuid.uuid4()).replace('-', '') # Simple mock hash
        }
        self.mock_chain.append(block)
        return block['hash']

    def record_yield_prediction(self, farm_id, predicted_yield, timestamp, app_data):
        if self.mock_mode:
            logging.info(f"Mocking blockchain record for prediction: Farm {farm_id}, Yield {predicted_yield}")
            data = {
                "type": "yield_prediction",
                "farm_id": farm_id,
                "predicted_yield": predicted_yield,
                "timestamp": timestamp.isoformat()
            }
            tx_hash = self._add_block(data)
            return tx_hash
        else:
            logging.warning("Blockchain not in mock mode. Real blockchain integration not implemented.")
            return None

    def record_actual_yield(self, farm_id, actual_yield, timestamp, app_data, prediction_tx_hash=None):
        if self.mock_mode:
            logging.info(f"Mocking blockchain record for actual yield: Farm {farm_id}, Actual {actual_yield}")
            data = {
                "type": "actual_yield",
                "farm_id": farm_id,
                "actual_yield": actual_yield,
                "timestamp": timestamp.isoformat(),
                "linked_prediction_tx": prediction_tx_hash
            }
            tx_hash = self._add_block(data)
            
            # Save actual yield to app_data
            actual_yield_record = {
                "id": str(uuid.uuid4()),
                "farm_id": farm_id,
                "actual_yield": actual_yield,
                "timestamp": timestamp,
                "linked_prediction_id": None # No direct link to prediction ID in JSON, would need to find it
            }
            app_data['actual_yields'].append(actual_yield_record)
            save_app_data(app_data)
            return tx_hash
        else:
            logging.warning("Blockchain not in mock mode. Real blockchain integration not implemented.")
            return None

    def log_shipment_event(self, shipment_id, farm_id, quantity_kg, origin_loc, dest_loc, timestamp, app_data):
        if self.mock_mode:
            logging.info(f"Mocking blockchain record for shipment: {shipment_id}")
            data = {
                "type": "shipment_event",
                "shipment_id": shipment_id,
                "farm_id": farm_id,
                "quantity_kg": quantity_kg,
                "origin_loc": origin_loc,
                "dest_loc": dest_loc,
                "timestamp": timestamp.isoformat()
            }
            tx_hash = self._add_block(data)

            # Save shipment to app_data
            shipment_record = {
                "id": str(uuid.uuid4()),
                "shipment_id": shipment_id,
                "farm_id": farm_id,
                "quantity_kg": quantity_kg,
                "origin_loc": origin_loc,
                "dest_loc": dest_loc,
                "timestamp": timestamp
            }
            app_data['shipments'].append(shipment_record)
            save_app_data(app_data)
            return tx_hash
        else:
            logging.warning("Blockchain not in mock mode. Real blockchain integration not implemented.")
            return None

    def get_farm_yield_history(self, farm_id, app_data):
        logging.info(f"Retrieving mock yield history for farm {farm_id}")
        # Filter predictions and actuals from app_data
        farm_predictions = [p for p in app_data['predictions'] if p['farm_id'] == farm_id]
        farm_actuals = [a for a in app_data['actual_yields'] if a['farm_id'] == farm_id]
        return {"predictions": farm_predictions, "actuals": farm_actuals}


# --- Main System Orchestrator Component ---
class AgriPredictXOrchestrator:
    def __init__(self):
        self.app_data = load_app_data() # Load data on init
        self.data_processor = DataProcessor()
        self.yield_predictor = YieldPredictor(MODEL_PATH)
        self.blockchain_client = BlockchainInterface({"mock_mode": True}) # Always in mock mode for this setup
        logging.info("AgriPredict-X Orchestrator initialized.")

    def initial_setup_and_training(self):
        logging.info("Performing initial setup and model training.")
        
        # Train the model (using synthetic data if no data is passed)
        mae, r2 = self.yield_predictor.train_model()
        logging.info(f"Initial model training complete. MAE: {mae:.2f}, R2: {r2:.2f}")

    def run_prediction_cycle(self, farm_id, lat, lon, current_date):
        """
        Orchestrates the data ingestion, ML prediction, and data recording for a farm.
        """
        logging.info(f"Starting prediction cycle for farm: {farm_id}")

        # 1. Data Ingestion & Preprocessing
        start_date_data = current_date - datetime.timedelta(days=180)
        
        weather_data = self.data_processor.fetch_weather_data(lat, lon, start_date_data, current_date)
        satellite_data = self.data_processor.fetch_satellite_data(bounding_box=[lon-0.1, lat-0.1, lon+0.1, lat+0.1], 
                                                                  start_date=start_date_data, end_date=current_date)
        
        # Load farm practices data (this will also create dummy if not exists)
        farm_practices_df = self.data_processor.load_farm_practices_data(farm_ids=[farm_id], app_data=self.app_data)
        
        # Soil data is part of farm practices for this mock
        soil_data_df = farm_practices_df # For consistency with old flow

        # Preprocess data to create features for the ML model
        prediction_input_df = self.data_processor.preprocess_data(
            weather_data, satellite_data, soil_data_df, farm_id, current_date, app_data=self.app_data
        )

        # 2. ML Prediction (will save prediction to app_data)
        try:
            predicted_yield = self.yield_predictor.make_prediction(prediction_input_df, app_data=self.app_data)[0]
            logging.info(f"Predicted yield for {farm_id}: {predicted_yield:.2f} tons/hectare")

            # 3. Record on Blockchain (mocked)
            self.blockchain_client.record_yield_prediction(farm_id, predicted_yield, current_date, app_data=self.app_data) 
            logging.info(f"Yield prediction for {farm_id} recorded in data and mocked on blockchain.")

        except Exception as e:
            logging.error(f"Error during ML prediction or data recording for {farm_id}: {e}")

    def run_supply_chain_logging_event(self, event_type, **kwargs):
        """
        Orchestrates logging a supply chain event to the data store and blockchain (mocked).
        """
        logging.info(f"Attempting to log supply chain event: {event_type}")
        try:
            if event_type == "shipment":
                self.blockchain_client.log_shipment_event(
                    shipment_id=kwargs['shipment_id'],
                    farm_id=kwargs['farm_id'],
                    quantity_kg=kwargs['quantity_kg'],
                    origin_loc=kwargs['origin_loc'],
                    dest_loc=kwargs['dest_loc'],
                    timestamp=kwargs['timestamp'],
                    app_data=self.app_data # Pass app_data
                )
                logging.info(f"Shipment {kwargs['shipment_id']} event logged successfully and mocked on blockchain.")
            elif event_type == "actual_yield":
                self.blockchain_client.record_actual_yield(
                    farm_id=kwargs['farm_id'],
                    actual_yield=kwargs['actual_yield'],
                    timestamp=kwargs['timestamp'],
                    app_data=self.app_data, # Pass app_data
                    prediction_tx_hash=kwargs.get('prediction_tx_hash')
                )
                logging.info(f"Actual yield for {kwargs['farm_id']} logged successfully and mocked on blockchain.")
            else:
                logging.warning(f"Unknown supply chain event type: {event_type}. No action taken.")
        except Exception as e:
            logging.error(f"Error logging {event_type} event: {e}")
            raise # Re-raise the exception after logging

    def populate_initial_dummy_data(self):
        """
        Populates the data store with an initial set of dummy farm data, predictions,
        actual yields, and shipments for demonstration purposes.
        This ensures the dashboard has data to display on first run.
        """
        logging.info("\n--- Populating initial dummy data for dashboard ---")
        
        farms_to_simulate = [
            {"id": "Dodoma_Farm_A", "lat": -6.1738, "lon": 35.7479, "initial_yield": 1.8},
            {"id": "Mbeya_Farm_B", "lat": -8.9038, "lon": 33.4568, "initial_yield": 2.2},
            {"id": "Morogoro_Farm_C", "lat": -6.8228, "lon": 37.6698, "initial_yield": 1.5},
            {"id": "Arusha_Farm_D", "lat": -3.3869, "lon": 36.6822, "initial_yield": 2.5},
            {"id": "Iringa_Farm_E", "lat": -7.7709, "lon": 35.6985, "initial_yield": 1.9},
            {"id": "Tabora_Farm_F", "lat": -5.0333, "lon": 32.8000, "initial_yield": 1.7},
        ]

        current_sim_date = datetime.datetime.now() # Starting simulation date

        for i, farm_info in enumerate(farms_to_simulate):
            logging.info(f"Simulating for {farm_info['id']} on {current_sim_date.strftime('%Y-%m-%d')}")
            # Ensure farm exists in app_data before processing
            existing_farm = next((f for f in self.app_data['farms'] if f['farm_id'] == farm_info['id']), None)
            if not existing_farm:
                # Add a full dummy farm record if it doesn't exist
                dummy_farm_data = {
                    "farm_id": farm_info['id'],
                    "crop_type": np.random.choice(['Maize', 'Rice', 'Wheat', 'Beans']),
                    "soil_type": np.random.choice(['Loam', 'Clay', 'Sand']),
                    "ph": round(np.random.uniform(5.5, 7.5), 1),
                    "nitrogen": round(np.random.uniform(50, 150), 2),
                    "phosphorus": round(np.random.uniform(30, 80), 2),
                    "planting_date": current_sim_date - datetime.timedelta(days=np.random.randint(60, 120)),
                    "fertilizer_applied": bool(np.random.choice([True, False])), # Ensure Python bool
                    "irrigation_method": np.random.choice(['Rainfed', 'Drip', 'Sprinkler']),
                    "previous_yield": farm_info['initial_yield'],
                    "last_updated": datetime.datetime.now()
                }
                self.app_data['farms'].append(dummy_farm_data)
                save_app_data(self.app_data)

            self.run_prediction_cycle(
                farm_id=farm_info['id'],
                lat=farm_info['lat'],
                lon=farm_info['lon'],
                current_date=current_sim_date
            )
            
            if i % 2 == 0 and i > 0: # Simulate an actual yield for some farms after a delay
                self.run_supply_chain_logging_event(
                    event_type="actual_yield",
                    farm_id=farm_info['id'],
                    actual_yield=round(farm_info['initial_yield'] * (0.9 + 0.2 * (i % 3)), 2),
                    timestamp=current_sim_date + datetime.timedelta(days=30),
                    prediction_tx_hash=f"mock_pred_tx_{farm_info['id']}_{current_sim_date.strftime('%Y%m%d')}"
                )
            
            current_sim_date += datetime.timedelta(days=np.random.randint(5, 15))

        shipment_counter = 1
        for i in range(len(farms_to_simulate) * 2): # Generate more shipments than farms
            farm_id = np.random.choice([f['id'] for f in farms_to_simulate])
            quantity = np.random.randint(500, 3000)
            origin = np.random.choice(['Dodoma Collection Point', 'Mbeya Hub', 'Morogoro Warehouse'])
            destination = np.random.choice(['Dar es Salaam Central Market', 'Arusha Distribution', 'Zanzibar Port'])
            shipment_time = datetime.datetime.now() + datetime.timedelta(days=np.random.randint(1, 60))
            
            self.run_supply_chain_logging_event(
                event_type="shipment",
                shipment_id=f"SHIP-{shipment_counter:03d}_{uuid.uuid4().hex[:8]}",
                farm_id=farm_id,
                quantity_kg=quantity,
                origin_loc=origin,
                dest_loc=destination,
                timestamp=shipment_time
            )
            shipment_counter += 1
            time.sleep(0.05) # Small delay

        logging.info("--- Initial dummy data population complete ---")

# --- Streamlit Dashboard App (Integrated) ---
def run_streamlit_app(orchestrator_instance):
    st.set_page_config(layout="wide")
    st.title("🌱 AgriPredict-X Dashboard for Tanzania 🇹🇿")

    st.markdown("""
    This dashboard provides insights into agricultural yield predictions and supply chain movements.
    Data displayed here is sourced from the local `app_data.json` file.
    """)

    # --- Data Fetching from App Data (with cache) ---
    @st.cache_data
    def fetch_prediction_data_for_dashboard(app_data_from_orchestrator):
        logging.info("Fetching prediction data for dashboard (or cache)...")
        # Ensure data is deep-copied or new dictionaries are created to avoid modifying cached dicts
        predictions = [p.copy() for p in app_data_from_orchestrator.get('predictions', [])]
        actuals = [a.copy() for a in app_data_from_orchestrator.get('actual_yields', [])]
        farm_data_records = [f.copy() for f in app_data_from_orchestrator.get('farms', [])]

        predictions_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()
        actuals_df = pd.DataFrame(actuals) if actuals else pd.DataFrame()
        farm_data_df = pd.DataFrame(farm_data_records) if farm_data_records else pd.DataFrame()

        if not predictions_df.empty:
            predictions_df = predictions_df.rename(columns={'timestamp': 'predicted_timestamp'})
            predictions_df['predicted_timestamp'] = pd.to_datetime(predictions_df['predicted_timestamp'])
            
            # Use farm_id for merging, ensure it's a string
            predictions_df['farm_id'] = predictions_df['farm_id'].astype(str)

            if not actuals_df.empty:
                actuals_df = actuals_df.rename(columns={'timestamp': 'actual_timestamp'})
                actuals_df['actual_timestamp'] = pd.to_datetime(actuals_df['actual_timestamp'])
                actuals_df['farm_id'] = actuals_df['farm_id'].astype(str)
                combined_df = predictions_df.merge(actuals_df, on='farm_id', how='left', suffixes=('_pred', '_actual'))
            else:
                combined_df = predictions_df.copy()
                combined_df['actual_yield'] = np.nan
                combined_df['actual_timestamp'] = pd.NaT
            
            # Add crop_type from farm_data_df
            if not farm_data_df.empty:
                farm_data_df['farm_id'] = farm_data_df['farm_id'].astype(str)
                combined_df = combined_df.merge(farm_data_df[['farm_id', 'crop_type']], on='farm_id', how='left')
            
            # Add a dummy region for demonstration
            if 'region' not in combined_df.columns:
                def assign_region(farm_id):
                    if farm_id.startswith("Dodoma"): return "Dodoma"
                    if farm_id.startswith("Mbeya"): return "Mbeya"
                    if farm_id.startswith("Morogoro"): return "Morogoro"
                    if farm_id.startswith("Arusha"): return "Arusha"
                    if farm_id.startswith("Iringa"): return "Iringa"
                    if farm_id.startswith("Tabora"): return "Tabora"
                    return np.random.choice(['Dodoma', 'Mbeya', 'Morogoro', 'Arusha', 'Iringa', 'Tabora'])

                combined_df['region'] = combined_df['farm_id'].apply(assign_region)
                
        else:
            combined_df = pd.DataFrame(columns=['farm_id', 'predicted_yield', 'actual_yield', 'predicted_timestamp', 'actual_timestamp', 'region', 'crop_type'])

        logging.info(f"Fetched dashboard prediction data. Contains {len(combined_df['farm_id'].unique()) if not combined_df.empty else 0} unique farms.")
        return combined_df

    @st.cache_data
    def fetch_shipment_data_for_dashboard(app_data_from_orchestrator):
        logging.info("Fetching shipment data for dashboard (or cache)...")
        shipments = [s.copy() for s in app_data_from_orchestrator.get('shipments', [])]
        shipments_df = pd.DataFrame(shipments) if shipments else pd.DataFrame()
        if not shipments_df.empty:
            shipments_df['timestamp'] = pd.to_datetime(shipments_df['timestamp'])
        logging.info(f"Fetched dashboard shipment data. Contains {len(shipments_df)} shipments.")
        return shipments_df

    # --- Functions to trigger actions ---
    def trigger_new_prediction_cycle_ui():
        # Clear specific caches affected by new data
        fetch_prediction_data_for_dashboard.clear()
        fetch_shipment_data_for_dashboard.clear()
        
        # Determine a new farm ID based on existing farms
        current_farm_count = len(orchestrator_instance.app_data['farms'])
        new_farm_id = f"NewFarm_{current_farm_count + 1:03d}"
        
        new_lat = np.random.uniform(-11.5, -1.0)
        new_lon = np.random.uniform(29.5, 40.5)
        current_time_for_new_data = datetime.datetime.now() + datetime.timedelta(minutes=np.random.randint(1, 60))
        
        with st.spinner(f"Running prediction cycle for {new_farm_id}..."):
            orchestrator_instance.run_prediction_cycle(
                farm_id=new_farm_id,
                lat=new_lat,
                lon=new_lon,
                current_date=current_time_for_new_data
            )
            orchestrator_instance.run_supply_chain_logging_event(
                event_type="actual_yield",
                farm_id=new_farm_id,
                actual_yield=round(np.random.uniform(1.0, 3.5), 2),
                timestamp=current_time_for_new_data + datetime.timedelta(days=np.random.randint(20, 40))
            )
            orchestrator_instance.run_supply_chain_logging_event(
                event_type="shipment",
                shipment_id=str(uuid.uuid4()),
                farm_id=new_farm_id,
                quantity_kg=np.random.randint(800, 2500),
                origin_loc=f"{new_farm_id.split('_')[0] if '_' in new_farm_id else 'Generated'} Local",
                dest_loc=np.random.choice(['Dar es Salaam Central Market', 'Arusha Distribution', 'Zanzibar Port']),
                timestamp=current_time_for_new_data + datetime.timedelta(days=np.random.randint(45, 70))
            )
        st.success(f"New prediction cycle for {new_farm_id} completed and data saved!")
        st.rerun()

    def retrain_ml_model_ui():
        # Clear all relevant caches before retraining
        st.cache_data.clear() # Clears both prediction and shipment data caches
        st.cache_resource.clear() # Clears orchestrator instance
        
        with st.spinner("Retraining ML Model... This might take a moment."):
            # Re-initialize orchestrator, which will trigger training via get_orchestrator
            # This calls the cached function which itself handles initial setup and training
            _ = get_orchestrator_cached() 
        st.success("ML Model retraining completed!")
        st.rerun()

    def clear_database_content_and_repopulate_ui():
        """
        Deletes all records from the in-memory app_data, then repopulates with initial dummy data,
        and saves to JSON.
        """
        with st.spinner("Clearing data content and repopulating..."):
            orchestrator_instance.app_data = { # Reset app_data directly in orchestrator
                "farms": [],
                "predictions": [],
                "actual_yields": [],
                "shipments": []
            }
            save_app_data(orchestrator_instance.app_data) # Save empty state
            logging.info("All existing data records deleted from app_data.json.")
            
            # Clear Streamlit caches so new data is fetched
            fetch_prediction_data_for_dashboard.clear()
            fetch_shipment_data_for_dashboard.clear()
            
            # Repopulate with dummy data
            orchestrator_instance.populate_initial_dummy_data()
            
        st.success("Data content cleared and re-populated with fresh dummy data (JSON structure preserved)!")
        st.rerun()

    def clear_all_data_and_reset_file_ui():
        """
        Deletes the app_data.json file and clears all caches,
        forcing a complete re-initialization.
        """
        if os.path.exists(APP_DATA_FILE):
            os.remove(APP_DATA_FILE)
            st.info(f"Deleted existing data file: {APP_DATA_FILE}.")
        
        st.cache_data.clear()
        st.cache_resource.clear() # Clear orchestrator cache (which calls get_orchestrator again on rerun)
        st.success("All caches cleared and data file reset! App will re-initialize with fresh data.")
        st.rerun()

    # --- Dashboard Layout ---
    # Pass the orchestrator's app_data to the fetching functions
    predictions_actuals_df = fetch_prediction_data_for_dashboard(orchestrator_instance.app_data)
    shipments_df = fetch_shipment_data_for_dashboard(orchestrator_instance.app_data)

    st.header("Yield Prediction Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Farms Monitored", len(predictions_actuals_df['farm_id'].unique()) if not predictions_actuals_df.empty else 0)
    with col2:
        st.metric("Avg Predicted Yield (t/ha)", f"{predictions_actuals_df['predicted_yield'].mean():.2f}" if not predictions_actuals_df.empty else "N/A")
    with col3:
        st.metric("Total Shipments Recorded", len(shipments_df) if not shipments_df.empty else 0)

    if not predictions_actuals_df.empty:
        st.subheader("Predicted vs. Actual Yields by Farm")
        selected_region = st.selectbox("Filter by Region", ['All'] + sorted(list(predictions_actuals_df['region'].unique())))
        if selected_region != 'All':
            filtered_predictions = predictions_actuals_df[predictions_actuals_df['region'] == selected_region]
        else:
            filtered_predictions = predictions_actuals_df

        if not filtered_predictions.empty:
            fig_yield_comp, ax_yield_comp = plt.subplots(figsize=(12, 6))
            plot_df = filtered_predictions[['farm_id', 'predicted_yield', 'actual_yield']].set_index('farm_id')
            plot_df.plot(kind='bar', ax=ax_yield_comp, width=0.8)
            ax_yield_comp.set_ylabel("Yield (tons/hectare)")
            ax_yield_comp.set_title(f"Predicted vs. Actual Yields in {selected_region}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_yield_comp)
        else:
            st.info("No prediction data for the selected region or farms in this region.")

        st.subheader("Yield Trends Over Time")
        # Ensure timestamp column is datetime before sorting
        predictions_actuals_df['predicted_timestamp'] = pd.to_datetime(predictions_actuals_df['predicted_timestamp'])
        predictions_actuals_df_sorted = predictions_actuals_df.sort_values('predicted_timestamp')

        fig_yield_trend, ax_yield_trend = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='predicted_timestamp', y='predicted_yield', data=predictions_actuals_df_sorted, label='Predicted', ax=ax_yield_trend)
        if 'actual_yield' in predictions_actuals_df_sorted.columns and not predictions_actuals_df_sorted['actual_yield'].isnull().all():
            sns.lineplot(x='predicted_timestamp', y='actual_yield', data=predictions_actuals_df_sorted.dropna(subset=['actual_yield']), label='Actual', ax=ax_yield_trend)
        ax_yield_trend.set_xlabel("Date")
        ax_yield_trend.set_ylabel("Yield (tons/hectare)")
        ax_yield_trend.set_title("Overall Yield Trend")
        ax_yield_trend.tick_params(axis='x', rotation=45)
        ax_yield_trend.legend()
        st.pyplot(fig_yield_trend)
    else:
        st.info("No yield prediction data available yet. Use the sidebar buttons to generate data.")

    st.header("Supply Chain Monitoring")
    if not shipments_df.empty:
        st.subheader("Shipment Volume by Destination")
        fig_shipment_dest, ax_shipment_dest = plt.subplots(figsize=(10, 5))
        shipments_df.groupby('dest_loc')['quantity_kg'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax_shipment_dest)
        ax_shipment_dest.set_ylabel("Total Quantity (kg)")
        ax_shipment_dest.set_xlabel("Destination Location")
        ax_shipment_dest.set_title("Total Shipment Volume by Destination")
        st.pyplot(fig_shipment_dest)

        st.subheader("Recent Shipments Log")
        st.dataframe(shipments_df.sort_values('timestamp', ascending=False).head(15))
    else:
        st.info("No shipment data available yet. Use the sidebar buttons to generate data.")

    st.sidebar.header("AgriPredict-X Control")
    st.sidebar.markdown("Use these buttons to interact with the system:")
    if st.sidebar.button("Trigger New Prediction Cycle", use_container_width=True):
        trigger_new_prediction_cycle_ui()
    if st.sidebar.button("Retrain ML Model", use_container_width=True):
        retrain_ml_model_ui()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Management:")
    if st.sidebar.button("Clear Data (Keep Structure)", help="Deletes all data records, then repopulates dummy data (keeps the JSON file structure).", use_container_width=True):
        clear_database_content_and_repopulate_ui()

    if st.sidebar.button("Clear All Data File & Reset", help="Deletes the 'app_data.json' file, clears all caches, and forces app to re-initialize with fresh dummy data.", use_container_width=True):
        clear_all_data_and_reset_file_ui()


# --- Global Orchestrator Caching for Streamlit ---
# This will be called by Streamlit when it runs the script, and its results cached.
@st.cache_resource(ttl=3600) # Cache for 1 hour, or until app restart/code change
def get_orchestrator_cached():
    logging.info("Initializing AgriPredictXOrchestrator and performing initial setup...")
    orch = AgriPredictXOrchestrator()
    orch.initial_setup_and_training()
    # Populate dummy data only if no predictions exist (first run scenario or after clearing data)
    if not orch.app_data['predictions']:
        logging.info("No existing prediction data found. Populating initial dummy data.")
        orch.populate_initial_dummy_data()
    return orch

# --- Main Entry Point ---
if __name__ == "__main__":
    # Check if the environment variable is set, indicating we are in Streamlit's context
    if os.getenv(STREAMLIT_APP_MODE_ENV) == "true":
        # If the environment variable is set, it means this instance was launched by
        # the initial Python script, and it should run as a Streamlit app.
        logging.info("AgriPredict-X Streamlit App: Detected Streamlit app mode via environment variable. Running application logic.")
        orchestrator_instance = get_orchestrator_cached()
        run_streamlit_app(orchestrator_instance)
    else:
        # If the environment variable is NOT set, it means `python main_system.py`
        # was executed directly. We need to launch Streamlit.
        logging.info("AgriPredict-X System: Direct Python execution detected. Launching Streamlit dashboard in a new process...")
        try:
            # Create a copy of the current environment and set the flag for the subprocess
            env = os.environ.copy()
            env[STREAMLIT_APP_MODE_ENV] = "true"

            # Construct the command to run Streamlit, ensuring it uses the current Python environment
            # os.path.abspath(__file__) provides the full path to the current script
            command = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]
            
            # Use subprocess.Popen to launch Streamlit in a non-blocking way.
            # Pass the modified environment to the new process.
            subprocess.Popen(command, env=env)
            logging.info(f"Streamlit dashboard launched with command: {' '.join(command)}")
            
            # Exit the current script immediately after launching the Streamlit process.
            # This prevents the original `python main_system.py` process from continuing
            # and avoids any potential conflicts or infinite loops.
            sys.exit(0)
            
        except FileNotFoundError:
            logging.error("Streamlit command 'streamlit' not found. Please ensure Streamlit is installed and in your system's PATH.")
            logging.info("You can install Streamlit using: pip install streamlit")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred while trying to launch Streamlit: {e}")
            sys.exit(1)