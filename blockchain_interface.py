import datetime
import hashlib # For simulating hashing data before "blockchain" storage

class BlockchainInterface:
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        print("BlockchainInterface initialized (mocked - no real blockchain connection).")
        # In a real system, you'd initialize Web3 or Hyperledger client here
        # self.w3 = Web3(Web3.HTTPProvider(self.config.get('rpc_url')))
        # self.contract = self.w3.eth.contract(...)

    def _simulate_transaction_hash(self, data):
        """Simulates generating a transaction hash."""
        return hashlib.sha256(str(data).encode()).hexdigest()[:16] # Shortened for display

    def record_yield_prediction(self, farm_id, predicted_yield, timestamp):
        """
        Mocks recording an AI-predicted yield on the blockchain.
        In a real system, this would be a smart contract call.
        """
        data_to_record = {
            "type": "yield_prediction",
            "farm_id": farm_id,
            "predicted_yield": round(predicted_yield, 2),
            "timestamp": timestamp.isoformat()
        }
        tx_hash = self._simulate_transaction_hash(data_to_record)
        print(f"MOCK BLOCKCHAIN: Yield prediction for farm {farm_id} ({predicted_yield:.2f} tons/ha) "
              f"recorded. Tx Hash: {tx_hash}")

    def record_actual_yield(self, farm_id, actual_yield, timestamp, prediction_tx_hash=None):
        """
        Mocks recording actual harvest yield and linking it to a prediction.
        """
        data_to_record = {
            "type": "actual_yield",
            "farm_id": farm_id,
            "actual_yield": round(actual_yield, 2),
            "timestamp": timestamp.isoformat(),
            "linked_prediction_tx": prediction_tx_hash
        }
        tx_hash = self._simulate_transaction_hash(data_to_record)
        print(f"MOCK BLOCKCHAIN: Actual yield for farm {farm_id} ({actual_yield:.2f} tons/ha) "
              f"recorded. Tx Hash: {tx_hash}")

    def log_shipment_event(self, shipment_id, farm_id, quantity_kg, origin_loc, dest_loc, timestamp):
        """
        Mocks logging a supply chain shipment event on the blockchain.
        """
        data_to_record = {
            "type": "shipment_event",
            "shipment_id": shipment_id,
            "farm_id": farm_id,
            "quantity_kg": quantity_kg,
            "origin_location": origin_loc,
            "destination_location": dest_loc,
            "timestamp": timestamp.isoformat()
        }
        tx_hash = self._simulate_transaction_hash(data_to_record)
        print(f"MOCK BLOCKCHAIN: Shipment {shipment_id} ({quantity_kg}kg) from {origin_loc} to {dest_loc} "
              f"logged. Tx Hash: {tx_hash}")

    def get_farm_yield_history(self, farm_id):
        """
        Mocks retrieving yield history for a specific farm.
        In a real system, this would query the blockchain state or events.
        """
        print(f"MOCK BLOCKCHAIN: Retrieving yield history for farm {farm_id}...")
        # Return dummy historical data for demonstration
        return {
            "F001": {"predictions": [1.8, 2.0], "actuals": [1.75, 2.05]},
            "Dodoma_Farm_A": {"predictions": [2.15], "actuals": []} # Example from orchestrator
        }.get(farm_id, {"predictions": [], "actuals": []})

# Example usage (for internal testing)
if __name__ == "__main__":
    client = BlockchainInterface()
    current_time = datetime.datetime.now()

    client.record_yield_prediction("DemoFarm_001", 2.5, current_time)
    client.record_actual_yield("DemoFarm_001", 2.6, current_time + datetime.timedelta(days=7), "prev_tx_hash_123")
    client.log_shipment_event("SHIP-XYZ-001", "DemoFarm_001", 1500, "Mbeya", "Dar es Salaam", current_time + datetime.timedelta(days=1))
    
    history = client.get_farm_yield_history("DemoFarm_001")
    print(f"\nSimulated History for DemoFarm_001: {history}")