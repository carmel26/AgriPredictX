# blockchain_interface.py
import datetime
import logging
import uuid # For generating unique IDs

# Assuming database models are available
from database import get_db, FarmData, Prediction, ActualYield, Shipment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BlockchainInterface:
    def __init__(self, config):
        self.mock_mode = config.get("mock_mode", True)
        if self.mock_mode:
            logging.info("BlockchainInterface initialized in mock mode. Data will be saved to DB only.")
        else:
            logging.warning("BlockchainInterface initialized in real mode (not implemented). Data will still be saved to DB.")

    def _generate_mock_tx_hash(self):
        """Generates a mock transaction hash."""
        return f"0x{uuid.uuid4().hex[:64]}"

    def record_yield_prediction(self, farm_id: str, predicted_yield: float, timestamp: datetime.datetime, db_session):
        """Records a yield prediction in the database and mocks a blockchain transaction."""
        try:
            # Save prediction to local database
            new_prediction = Prediction(
                farm_id=farm_id,
                predicted_yield=predicted_yield,
                timestamp=timestamp
            )
            db_session.add(new_prediction)
            db_session.commit() # <--- ADDED/ENSURED THIS LINE
            logging.info(f"Prediction for farm {farm_id} saved to DB with ID: {new_prediction.id}")

            # Mock blockchain interaction
            tx_hash = self._generate_mock_tx_hash()
            logging.info(f"Mock blockchain: Recorded yield prediction for farm {farm_id}. Tx Hash: {tx_hash}")
            return tx_hash
        except Exception as e:
            db_session.rollback() # Rollback in case of error
            logging.error(f"Failed to record yield prediction for farm {farm_id}: {e}")
            raise # Re-raise the exception after logging

    def record_actual_yield(self, farm_id: str, actual_yield: float, timestamp: datetime.datetime, db_session, prediction_tx_hash: str = None):
        """Records an actual yield in the database and mocks a blockchain transaction."""
        try:
            # Save actual yield to local database
            # Attempt to link to a prediction if a recent one exists for the farm
            linked_prediction = db_session.query(Prediction).filter(
                Prediction.farm_id == farm_id
            ).order_by(Prediction.timestamp.desc()).first()

            new_actual = ActualYield(
                farm_id=farm_id,
                actual_yield=actual_yield,
                timestamp=timestamp,
                linked_prediction_id=linked_prediction.id if linked_prediction else None
            )
            db_session.add(new_actual)
            db_session.commit() # <--- ADDED/ENSURED THIS LINE
            logging.info(f"Actual yield for farm {farm_id} saved to DB with ID: {new_actual.id}")

            # Mock blockchain interaction
            tx_hash = self._generate_mock_tx_hash()
            logging.info(f"Mock blockchain: Recorded actual yield for farm {farm_id}. Tx Hash: {tx_hash}")
            return tx_hash
        except Exception as e:
            db_session.rollback()
            logging.error(f"Failed to record actual yield for farm {farm_id}: {e}")
            raise

    def log_shipment_event(self, shipment_id: str, farm_id: str, quantity_kg: float, origin_loc: str, dest_loc: str, timestamp: datetime.datetime, db_session):
        """Logs a shipment event in the database and mocks a blockchain transaction."""
        try:
            # Save shipment to local database
            new_shipment = Shipment(
                shipment_id=shipment_id,
                farm_id=farm_id,
                quantity_kg=quantity_kg,
                origin_loc=origin_loc,
                dest_loc=dest_loc,
                timestamp=timestamp
            )
            db_session.add(new_shipment)
            db_session.commit() # <--- ADDED/ENSURED THIS LINE
            logging.info(f"Shipment {shipment_id} from farm {farm_id} saved to DB with ID: {new_shipment.id}")

            # Mock blockchain interaction
            tx_hash = self._generate_mock_tx_hash()
            logging.info(f"Mock blockchain: Logged shipment {shipment_id}. Tx Hash: {tx_hash}")
            return tx_hash
        except Exception as e:
            db_session.rollback()
            logging.error(f"Failed to log shipment {shipment_id}: {e}")
            raise

    def get_farm_yield_history(self, farm_id: str, db_session):
        """Retrieves yield history for a specific farm from the database."""
        predictions = db_session.query(Prediction).filter_by(farm_id=farm_id).order_by(Prediction.timestamp.asc()).all()
        actuals = db_session.query(ActualYield).filter_by(farm_id=farm_id).order_by(ActualYield.timestamp.asc()).all()

        pred_data = [{"timestamp": p.timestamp.isoformat(), "predicted_yield": p.predicted_yield} for p in predictions]
        actual_data = [{"timestamp": a.timestamp.isoformat(), "actual_yield": a.actual_yield} for a in actuals]

        return {"predictions": pred_data, "actuals": actual_data}

    def get_all_shipments(self, db_session):
        """Retrieves all shipment records from the database."""
        shipments = db_session.query(Shipment).order_by(Shipment.timestamp.asc()).all()
        shipment_data = [{
            "shipment_id": s.shipment_id,
            "farm_id": s.farm_id,
            "quantity_kg": s.quantity_kg,
            "origin_loc": s.origin_loc,
            "dest_loc": s.dest_loc,
            "timestamp": s.timestamp.isoformat()
        } for s in shipments]
        return shipment_data