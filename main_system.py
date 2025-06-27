import numpy as np
import datetime
import time
import uuid
import os
import json
import logging
import joblib # For loading/saving the ML model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration and Constants ---
APP_DATA_FILE = "app_data.json"
MODEL_PATH = "models/random_forest_model.joblib"
# Ensure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

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
        # Generate dummy weather data as a list of dictionaries
        dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        weather_data = []
        for date_val in dates:
            weather_data.append({
                'date': date_val,
                'temperature_avg': np.random.uniform(20, 30),
                'humidity_avg': np.random.uniform(60, 90),
                'precipitation': np.random.uniform(0, 10)
            })
        return weather_data

    def fetch_satellite_data(self, bounding_box, start_date, end_date):
        logging.info(f"Fetching mock satellite data for {bounding_box} from {start_date} to {end_date}")
        # Generate dummy satellite data (e.g., NDVI) as a list of dictionaries
        dates = [start_date + datetime.timedelta(days=i * 7) for i in range(((end_date - start_date).days // 7) + 1)]
        satellite_data = []
        for date_val in dates:
            satellite_data.append({
                'date': date_val,
                'ndvi': np.random.uniform(0.3, 0.8)
            })
        return satellite_data

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
            return [dummy_farm_data] # Return as list of dicts
        
        return existing_farms


    def load_soil_data(self, farm_ids, app_data):
        logging.info(f"Loading soil data for {farm_ids}")
        # Soil data is typically static or part of farm practices data for this simulation
        return self.load_farm_practices_data(farm_ids, app_data) # Re-use farm practices for basic soil info

    def preprocess_data(self, weather_data, satellite_data, farm_data, current_farm_id, current_date, app_data):
        logging.info(f"Preprocessing data for farm: {current_farm_id}")
        
        # Simple aggregation for weather and satellite data
        # In a real scenario, this would involve more complex time-series processing.
        # Here, we'll just take the average or latest values.

        latest_env_data = {}
        if weather_data:
            latest_weather = weather_data[-1] # Take the latest available
            latest_env_data['temperature_avg'] = latest_weather['temperature_avg']
            latest_env_data['humidity_avg'] = latest_weather['humidity_avg']
            latest_env_data['precipitation'] = latest_weather['precipitation']
            latest_env_data['gdd'] = max(0, latest_weather['temperature_avg'] - 10) # Simple GDD calculation

        if satellite_data:
            latest_satellite = satellite_data[-1] # Take the latest available
            latest_env_data['ndvi'] = latest_satellite['ndvi']

        # Find the specific farm's data
        farm_features = None
        for farm_rec in farm_data:
            if farm_rec['farm_id'] == current_farm_id:
                farm_features = farm_rec
                break

        if farm_features is None:
            logging.error(f"Farm data not found for {current_farm_id} during preprocessing.")
            raise ValueError(f"Farm data not found for {current_farm_id}.")

        # Create a single feature vector (dictionary) for prediction
        features_dict = {
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

        # Include categorical features as separate dictionary entries for now
        # The YieldPredictor will handle mapping these to numerical values (e.g., one-hot encoding if needed)
        features_dict['crop_type'] = farm_features.get('crop_type', 'Maize')
        features_dict['soil_type'] = farm_features.get('soil_type', 'Loam')
        features_dict['irrigation_method'] = farm_features.get('irrigation_method', 'Rainfed')

        # Return a list containing this single feature dictionary
        return [features_dict]


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
        # Example categories for one-hot encoding if needed (for internal use by model)
        self.crop_types = ['Maize', 'Rice', 'Wheat', 'Beans']
        self.soil_types = ['Loam', 'Clay', 'Sand']
        self.irrigation_methods = ['Rainfed', 'Drip', 'Sprinkler']
        
        # To handle categorical features for prediction, we need consistent encoding.
        # A simple approach for this mock: map them to indices or use one-hot if building a pipeline.
        # For this basic RF model, we will only use numerical features from the `feature_columns` list directly.
        # If the categorical features were needed by the model, they would need explicit encoding steps here
        # or as part of a scikit-learn Pipeline.

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

    def train_model(self, data=None):
        logging.info("Training ML model...")
        
        if data is None or not data:
            logging.info("Generating synthetic data for model training.")
            num_samples = 100
            training_data_list = []
            for _ in range(num_samples):
                training_data_list.append({
                    'temperature_avg': np.random.uniform(20, 30),
                    'humidity_avg': np.random.uniform(60, 90),
                    'precipitation': np.random.uniform(0, 10),
                    'ndvi': np.random.uniform(0.3, 0.8),
                    'gdd': np.random.uniform(10, 25),
                    'ph': np.random.uniform(5.5, 7.5),
                    'nitrogen': np.random.uniform(50, 150),
                    'phosphorus': np.random.uniform(30, 80),
                    'fertilizer_applied': np.random.randint(0, 2),
                    'previous_yield': np.random.uniform(1.0, 3.0),
                    'days_since_planting': np.random.randint(60, 180),
                    'crop_type': np.random.choice(self.crop_types),
                    'soil_type': np.random.choice(self.soil_types),
                    'irrigation_method': np.random.choice(self.irrigation_methods),
                    'yield': np.random.uniform(1.5, 4.0) # Target variable
                })
            data = training_data_list
        
        # Prepare data for Scikit-learn (numerical features only for this basic RF)
        # Extract features and target into NumPy arrays
        X_list = []
        y_list = []
        for record in data:
            row_features = [record[col] for col in self.feature_columns]
            X_list.append(row_features)
            y_list.append(record['yield'])
        
        X = np.array(X_list)
        y = np.array(y_list)

        # Ensure there's enough data for splitting
        if len(X) < 2:
            logging.warning("Not enough data to perform train-test split. Skipping training.")
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self._save_model()

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model training complete. MAE: {mae:.2f}, R2: {r2:.2f}")
        return mae, r2

    def make_prediction(self, input_features_list, app_data):
        logging.info("Making prediction...")
        if self.model is None:
            logging.error("Model not trained. Training initial model.")
            self.train_model() # Train if not already trained

        if not input_features_list:
            raise ValueError("Input features list is empty.")
            
        # Assuming input_features_list contains a single dictionary for prediction
        input_features_dict = input_features_list[0] 
        
        # Extract features into a NumPy array in the correct order
        X_predict_row = [input_features_dict[col] for col in self.feature_columns]
        X_predict = np.array([X_predict_row]) # Reshape for single prediction

        predicted_yield = self.model.predict(X_predict)
        
        # Save prediction to app_data
        prediction_record = {
            "id": str(uuid.uuid4()),
            "farm_id": input_features_dict['farm_id'], # Assuming farm_id is in input_features_dict
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
        farm_practices_data = self.data_processor.load_farm_practices_data(farm_ids=[farm_id], app_data=self.app_data)
        
        # Soil data is part of farm practices for this mock
        soil_data = farm_practices_data # For consistency with old flow

        # Preprocess data to create features for the ML model
        prediction_input_features = self.data_processor.preprocess_data(
            weather_data, satellite_data, soil_data, farm_id, current_date, app_data=self.app_data
        )

        # 2. ML Prediction (will save prediction to app_data)
        try:
            predicted_yield = self.yield_predictor.make_prediction(prediction_input_features, app_data=self.app_data)[0]
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
        This ensures the system has data to work with on first run.
        """
        logging.info("\n--- Populating initial dummy data ---")
        
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

# --- Main Entry Point ---
if __name__ == "__main__":
    logging.info("AgriPredict-X Backend System: Starting up...")
    
    # Initialize the orchestrator
    orchestrator_instance = AgriPredictXOrchestrator()
    
    # Perform initial setup and model training
    orchestrator_instance.initial_setup_and_training()
    
    # Populate dummy data if no predictions exist (first run scenario or after clearing data)
    if not orchestrator_instance.app_data['predictions']:
        logging.info("No existing prediction data found. Populating initial dummy data.")
        orchestrator_instance.populate_initial_dummy_data()

    logging.info("AgriPredict-X Backend System: Initialization complete. Data is processed and saved to app_data.json.")
    logging.info("You can inspect the 'app_data.json' file to see the generated data.")
    # The script will now simply exit after completing its tasks.