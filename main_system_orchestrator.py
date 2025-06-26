import pandas as pd
import datetime
import os
import json
import logging

# Import modules from our project
from data_ingestion_preprocessing import DataProcessor
from ml_model_training_prediction import YieldPredictor
from blockchain_interface import BlockchainInterface

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriPredictXOrchestrator:
    def __init__(self, config_path="config.json"):
        # First, load or create the configuration
        self.config = self._load_or_create_config(config_path)
        
        # Then, use the loaded/created configuration to initialize other components
        self.data_processor = DataProcessor(self.config.get('data_processor', {}))
        self.yield_predictor = YieldPredictor(self.config.get('model_path', 'models/random_forest_model.joblib'))
        self.blockchain_client = BlockchainInterface(self.config.get('blockchain', {}))
        
        logging.info("AgriPredict-X Orchestrator initialized.")

    def _load_or_create_config(self, config_path):
        """Loads configuration from a JSON file or creates a default one if not found."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logging.warning(f"Config file not found at {config_path}. Creating a dummy one.")
            dummy_config = {
                "data_processor": {
                    "area_of_interest": "Tanzania" # Example config for data processor
                },
                "model_path": "models/random_forest_model.joblib",
                "data_storage_dir": "data/",
                "blockchain": {
                    "mock_mode": True # Indicates we are in mock mode for blockchain
                }
            }
            
            # Create directories needed for the project using the dummy_config
            # This part was causing the error as self.config was not yet assigned
            os.makedirs(os.path.dirname(dummy_config.get('model_path', 'models/random_forest_model.joblib')), exist_ok=True)
            os.makedirs(dummy_config.get('data_storage_dir', 'data/'), exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(dummy_config, f, indent=4)
            return dummy_config

    def initial_setup_and_training(self):
        """
        Performs initial setup, including training the ML model.
        This should be run once or when the model needs retraining.
        """
        logging.info("Performing initial setup and model training.")
        # The YieldPredictor's train_model method will generate synthetic data if none is provided.
        mae, r2 = self.yield_predictor.train_model()
        logging.info(f"Initial model training complete. MAE: {mae:.2f}, R2: {r2:.2f}")

    def run_prediction_cycle(self, farm_id, lat, lon, current_date):
        """
        Orchestrates the data ingestion, ML prediction, and blockchain recording for a farm.
        """
        logging.info(f"Starting prediction cycle for farm: {farm_id}")

        # 1. Data Ingestion & Preprocessing
        # Simulate relevant date ranges for data fetching
        start_date_data = current_date - datetime.timedelta(days=180) # e.g., last 6 months of historical data
        
        # Ingest simulated data for the specific farm's context
        weather_data = self.data_processor.fetch_weather_data(lat, lon, start_date_data, current_date)
        satellite_data = self.data_processor.fetch_satellite_data(bounding_box=[lon-0.1, lat-0.1, lon+0.1, lat+0.1], 
                                                                  start_date=start_date_data, end_date=current_date)
        
        # For soil and farm practices, we assume some base data for demonstration.
        # In a real system, this would be pulled from a database or farmer input system.
        soil_data = self.data_processor.load_soil_data(farm_ids=[farm_id])
        farm_practices_data = self.data_processor.load_farm_practices_data(farm_ids=[farm_id])

        # Preprocess data to create features for the ML model
        # This function encapsulates logic to align and aggregate data for the ML input
        prediction_input_df = self.data_processor.preprocess_data(
            weather_data, satellite_data, soil_data, farm_practices_data, farm_id, current_date
        )

        # 2. ML Prediction
        try:
            predicted_yield = self.yield_predictor.make_prediction(prediction_input_df)[0]
            logging.info(f"Predicted yield for {farm_id}: {predicted_yield:.2f} tons/hectare")

            # 3. Record on Blockchain (mocked)
            self.blockchain_client.record_yield_prediction(farm_id, predicted_yield, current_date)
            logging.info(f"Yield prediction for {farm_id} would be securely recorded.")

        except Exception as e:
            logging.error(f"Error during ML prediction or blockchain recording for {farm_id}: {e}")

    def run_supply_chain_logging_event(self, event_type, **kwargs):
        """
        Orchestrates logging a supply chain event to the blockchain (mocked).
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
                    timestamp=kwargs['timestamp']
                )
                logging.info(f"Shipment {kwargs['shipment_id']} event would be logged successfully.")
            elif event_type == "actual_yield":
                self.blockchain_client.record_actual_yield(
                    farm_id=kwargs['farm_id'],
                    actual_yield=kwargs['actual_yield'],
                    timestamp=kwargs['timestamp'],
                    prediction_tx_hash=kwargs.get('prediction_tx_hash') # Optional: link to a prediction
                )
                logging.info(f"Actual yield for {kwargs['farm_id']} would be logged successfully.")
            else:
                logging.warning(f"Unknown supply chain event type: {event_type}. No action taken.")
        except Exception as e:
            logging.error(f"Error logging {event_type} event: {e}")

# Main execution block
if __name__ == "__main__":
    orchestrator = AgriPredictXOrchestrator()

    # --- Step 1: Initial Model Training (Run once or periodically) ---
    orchestrator.initial_setup_and_training()

    # --- Step 2: Run a Prediction Cycle for a specific farm ---
    # Example: Simulating a farm in Dodoma region, Tanzania (approximate lat/lon)
    farm_id_to_predict = "Dodoma_Farm_A"
    farm_lat = -6.1738
    farm_lon = 35.7479
    current_sim_date = datetime.datetime.now()

    orchestrator.run_prediction_cycle(
        farm_id=farm_id_to_predict,
        lat=farm_lat,
        lon=farm_lon,
        current_date=current_sim_date
    )

    # --- Step 3: Simulate and Log a Supply Chain Event (e.g., a shipment) ---
    orchestrator.run_supply_chain_logging_event(
        event_type="shipment",
        shipment_id="SHIP-DOD-DAR-001",
        farm_id=farm_id_to_predict,
        quantity_kg=1500, # kg of produce
        origin_loc="Dodoma Collection Point",
        dest_loc="Dar es Salaam Central Market",
        timestamp=current_sim_date + datetime.timedelta(days=1) # Next day
    )

    # --- Step 4: Simulate and Log an Actual Yield after harvest ---
    orchestrator.run_supply_chain_logging_event(
        event_type="actual_yield",
        farm_id=farm_id_to_predict,
        actual_yield=2.05, # Actual yield observed
        timestamp=current_sim_date + datetime.timedelta(days=30), # After a month
        prediction_tx_hash="some_mock_tx_hash_from_step2" # Link to a previous prediction
    )

    # --- Step 5: Retrieve Simulated History (conceptual) ---
    print("\n--- Retrieving Simulated Farm History ---")
    history = orchestrator.blockchain_client.get_farm_yield_history(farm_id_to_predict)
    print(f"History for {farm_id_to_predict}:")
    print(f"  Predicted Yields: {history['predictions']}")
    print(f"  Actual Yields: {history['actuals']}")

    logging.info("AgriPredict-X demonstration complete. Check print statements for mocked outputs.")