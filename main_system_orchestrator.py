import pandas as pd
import datetime
import os
import json
import logging
import time
import numpy as np
import uuid # <--- THIS IMPORT IS CRUCIAL

# Import modules from our project
from data_ingestion_preprocessing import DataProcessor
from ml_model_training_prediction import YieldPredictor
from blockchain_interface import BlockchainInterface

# Import database functions and models
from database import create_db_tables, get_db, FarmData, Prediction, ActualYield, Shipment

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriPredictXOrchestrator:
    def __init__(self, config_path="config.json"):
        self.config = self._load_or_create_config(config_path)
        
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
            
            os.makedirs(os.path.dirname(dummy_config.get('model_path', 'models/random_forest_model.joblib')), exist_ok=True)
            os.makedirs(dummy_config.get('data_storage_dir', 'data/'), exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(dummy_config, f, indent=4)
            return dummy_config

    def initial_setup_and_training(self):
        """
        Performs initial setup, including creating database tables and training the ML model.
        This should be run once or when the model needs retraining.
        """
        logging.info("Performing initial setup and model training.")
        
        # 1. Create database tables if they don't exist
        create_db_tables()
        logging.info("Database tables ensured.")

        # 2. Train the model (using synthetic data if no data is passed)
        mae, r2 = self.yield_predictor.train_model()
        logging.info(f"Initial model training complete. MAE: {mae:.2f}, R2: {r2:.2f}")

    def run_prediction_cycle(self, farm_id, lat, lon, current_date):
        """
        Orchestrates the data ingestion, ML prediction, and database/blockchain recording for a farm.
        """
        logging.info(f"Starting prediction cycle for farm: {farm_id}")

        with get_db() as db: # Obtain a database session
            # 1. Data Ingestion & Preprocessing
            start_date_data = current_date - datetime.timedelta(days=180)
            
            weather_data = self.data_processor.fetch_weather_data(lat, lon, start_date_data, current_date)
            satellite_data = self.data_processor.fetch_satellite_data(bounding_box=[lon-0.1, lat-0.1, lon+0.1, lat+0.1], 
                                                                      start_date=start_date_data, end_date=current_date)
            
            # Load and/or save farm practices data to DB
            farm_practices_data = self.data_processor.load_farm_practices_data(farm_ids=[farm_id], db_session=db)
            
            # Load and/or save soil data to DB
            soil_data = self.data_processor.load_soil_data(farm_ids=[farm_id], db_session=db)

            # Preprocess data to create features for the ML model
            prediction_input_df = self.data_processor.preprocess_data(
                weather_data, satellite_data, soil_data, farm_practices_data, farm_id, current_date, db_session=db
            )

            # 2. ML Prediction
            try:
                # Pass the db session to make_prediction for saving the prediction
                predicted_yield = self.yield_predictor.make_prediction(prediction_input_df, db_session=db)[0]
                logging.info(f"Predicted yield for {farm_id}: {predicted_yield:.2f} tons/hectare")

                # 3. Record on Blockchain (mocked) and save to DB
                # Pass the db session to blockchain client for saving the 'blockchain' event
                self.blockchain_client.record_yield_prediction(farm_id, predicted_yield, current_date, db_session=db) 
                logging.info(f"Yield prediction for {farm_id} recorded in database and mocked on blockchain.")

            except Exception as e:
                logging.error(f"Error during ML prediction or database/blockchain recording for {farm_id}: {e}")

    def run_supply_chain_logging_event(self, event_type, **kwargs):
        """
        Orchestrates logging a supply chain event to the database and blockchain (mocked).
        """
        logging.info(f"Attempting to log supply chain event: {event_type}")
        with get_db() as db: # Obtain a database session
            try:
                if event_type == "shipment":
                    self.blockchain_client.log_shipment_event(
                        shipment_id=kwargs['shipment_id'],
                        farm_id=kwargs['farm_id'],
                        quantity_kg=kwargs['quantity_kg'],
                        origin_loc=kwargs['origin_loc'],
                        dest_loc=kwargs['dest_loc'],
                        timestamp=kwargs['timestamp'],
                        db_session=db # Pass db session
                    )
                    logging.info(f"Shipment {kwargs['shipment_id']} event logged successfully in DB and mocked on blockchain.")
                elif event_type == "actual_yield":
                    self.blockchain_client.record_actual_yield(
                        farm_id=kwargs['farm_id'],
                        actual_yield=kwargs['actual_yield'],
                        timestamp=kwargs['timestamp'],
                        db_session=db, # Pass db session
                        prediction_tx_hash=kwargs.get('prediction_tx_hash') # Optional: link to a previous prediction
                    )
                    logging.info(f"Actual yield for {kwargs['farm_id']} logged successfully in DB and mocked on blockchain.")
                else:
                    logging.warning(f"Unknown supply chain event type: {event_type}. No action taken.")
            except Exception as e:
                db.rollback() # Ensure rollback on error
                logging.error(f"Error logging {event_type} event: {e}")
                raise # Re-raise the exception after logging

    def populate_initial_dummy_data(self):
        """
        Populates the database with an initial set of dummy farm data, predictions,
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
                shipment_id=f"SHIP-{shipment_counter:03d}_{uuid.uuid4().hex[:8]}", # Ensure unique ID
                farm_id=farm_id,
                quantity_kg=quantity,
                origin_loc=origin,
                dest_loc=destination,
                timestamp=shipment_time
            )
            shipment_counter += 1
            time.sleep(0.05) # Small delay

        logging.info("--- Initial dummy data population complete ---")

# Main execution block for standalone running (e.g., for testing or initial setup)
if __name__ == "__main__":
    orchestrator = AgriPredictXOrchestrator()
    orchestrator.initial_setup_and_training()
    orchestrator.populate_initial_dummy_data()

    # Example of retrieving history for one farm
    farm_id_to_check_history = "Dodoma_Farm_A"
    print(f"\n--- Retrieving Farm History from Database for {farm_id_to_check_history} ---")
    with get_db() as db:
        history = orchestrator.blockchain_client.get_farm_yield_history(farm_id_to_check_history, db_session=db)
    print(f"  Predicted Yields: {history['predictions']}")
    print(f"  Actual Yields: {history['actuals']}")

    logging.info("AgriPredict-X standalone demonstration complete.")