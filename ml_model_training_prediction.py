import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from sqlalchemy.orm import Session
from database import Prediction # Import Prediction model

class YieldPredictor:
    def __init__(self, model_path="models/random_forest_model.joblib"):
        self.model = None
        self.model_path = model_path
        self.features = None

    def _generate_synthetic_training_data(self, num_samples=100):
        """Generates synthetic training data for demonstration."""
        print(f"Generating {num_samples} synthetic training samples...")
        data = {
            'farm_id': [f'FARM_{i:03d}' for i in range(num_samples)],
            'planting_month': np.random.randint(1, 12, num_samples),
            'avg_temp_growth_period': np.random.uniform(20, 35, num_samples),
            'total_precip_growth_period': np.random.uniform(100, 800, num_samples),
            'avg_ndvi_growth_period': np.random.uniform(0.3, 0.8, num_samples),
            'soil_ph': np.random.uniform(5.5, 7.5, num_samples),
            'soil_nitrogen': np.random.uniform(10, 60, num_samples),
            'previous_yield': np.random.uniform(0.5, 3.5, num_samples),
        }
        df = pd.DataFrame(data)
        
        df['yield'] = (
            1.5 
            + df['avg_temp_growth_period'] * 0.05 
            + df['total_precip_growth_period'] * 0.002 
            + df['avg_ndvi_growth_period'] * 2.0 
            - df['soil_ph'] * 0.1 
            + df['previous_yield'] * 0.3
            + np.random.normal(0, 0.2, num_samples)
        )
        df['yield'] = np.clip(df['yield'], 0.1, 5.0)
        return df

    def train_model(self, data_df=None, target_column='yield', test_size=0.2, random_state=42):
        """Trains the machine learning model."""
        print("Starting model training...")
        if data_df is None:
            data_df = self._generate_synthetic_training_data()

        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        self.features = [col for col in data_df.columns if col not in ['farm_id', target_column]]
        X = data_df[self.features]
        y = data_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model Training Complete. MAE: {mae:.2f}, R-squared: {r2:.2f}")

        self.save_model()
        return mae, r2

    def make_prediction(self, new_data_df, db_session: Session = None):
        """
        Makes yield predictions on new, unseen data and optionally saves to DB.
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise ValueError("Model not trained or loaded. Please train or load a model first.")

        if self.features:
            for feature in self.features:
                if feature not in new_data_df.columns:
                    new_data_df[feature] = 0
            new_data_df = new_data_df[self.features]

        print("Making yield predictions...")
        predictions = self.model.predict(new_data_df)

        if db_session and not new_data_df.empty:
            for index, row in new_data_df.iterrows():
                farm_id = row['farm_id'] if 'farm_id' in row else "UNKNOWN_FARM" # Assuming farm_id exists
                predicted_yield_value = predictions[index]
                
                new_prediction_entry = Prediction(
                    farm_id=farm_id,
                    predicted_yield=float(predicted_yield_value), # Ensure float type
                    timestamp=datetime.datetime.now()
                )
                db_session.add(new_prediction_entry)
                try:
                    db_session.commit()
                    db_session.refresh(new_prediction_entry)
                    print(f"Prediction for {farm_id} saved to database. ID: {new_prediction_entry.id}")
                except Exception as e:
                    db_session.rollback()
                    print(f"Error saving prediction for {farm_id} to DB: {e}")

        return predictions

    def save_model(self):
        """Saves the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'features': self.features}, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads a pre-trained model from disk."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.features = data['features']
            print(f"Model and features loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}. Please train a model first.")

# Example usage (for internal testing) - Now requires a dummy DB session if run directly
if __name__ == "__main__":
    from database import get_db, create_db_tables
    create_db_tables() # Ensure tables exist for testing

    predictor = YieldPredictor()
    predictor.train_model()

    new_farm_data = pd.DataFrame({
        'farm_id': ['F006'],
        'planting_month': [4],
        'avg_temp_growth_period': [27],
        'total_precip_growth_period': [360],
        'avg_ndvi_growth_period': [0.68],
        'soil_ph': [6.9],
        'soil_nitrogen': [45],
        'previous_yield': [1.65]
    })
    
    with get_db() as db:
        predicted_yields = predictor.make_prediction(new_farm_data, db_session=db)
        print(f"\nPredicted yield for F006: {predicted_yields[0]:.2f} tons/hectare")