import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # For saving/loading models
import os

class YieldPredictor:
    def __init__(self, model_path="models/random_forest_model.joblib"):
        self.model = None
        self.model_path = model_path
        self.features = None # To store feature names used during training

    def _generate_synthetic_training_data(self, num_samples=100):
        """
        Generates synthetic training data for demonstration.
        In a real system, this would be loaded from preprocessed data files.
        """
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
        
        # Create a synthetic 'yield' target variable based on features
        # This relationship is arbitrary for demonstration
        df['yield'] = (
            1.5 
            + df['avg_temp_growth_period'] * 0.05 
            + df['total_precip_growth_period'] * 0.002 
            + df['avg_ndvi_growth_period'] * 2.0 
            - df['soil_ph'] * 0.1 
            + df['previous_yield'] * 0.3
            + np.random.normal(0, 0.2, num_samples) # Add some noise
        )
        df['yield'] = np.clip(df['yield'], 0.1, 5.0) # Ensure yield is realistic
        return df

    def train_model(self, data_df=None, target_column='yield', test_size=0.2, random_state=42):
        """
        Trains the machine learning model.
        If data_df is None, synthetic data will be generated.
        """
        print("Starting model training...")
        if data_df is None:
            data_df = self._generate_synthetic_training_data()

        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        self.features = [col for col in data_df.columns if col not in ['farm_id', target_column]]
        X = data_df[self.features]
        y = data_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Using RandomForestRegressor as an example
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model Training Complete. MAE: {mae:.2f}, R-squared: {r2:.2f}")

        self.save_model()
        return mae, r2

    def make_prediction(self, new_data_df):
        """
        Makes yield predictions on new, unseen data.
        new_data_df: DataFrame with the same features as used for training.
        """
        if self.model is None:
            self.load_model()
            if self.model is None: # If still None after attempting to load
                raise ValueError("Model not trained or loaded. Please train or load a model first.")

        # Ensure new_data_df has the same columns and order as features used for training
        if self.features: # Only if features are defined (i.e., model was trained/loaded)
            # Add missing features with default (e.g., 0) if they are not in new_data_df
            for feature in self.features:
                if feature not in new_data_df.columns:
                    new_data_df[feature] = 0 # Or a sensible default/imputation strategy
            new_data_df = new_data_df[self.features] # Reorder columns

        print("Making yield predictions...")
        predictions = self.model.predict(new_data_df)
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

# Example usage (for internal testing)
if __name__ == "__main__":
    predictor = YieldPredictor()
    
    # Train the model using synthetic data
    predictor.train_model()

    # Simulate new data for prediction (this would come from DataProcessor)
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
    
    predicted_yields = predictor.make_prediction(new_farm_data)
    print(f"\nPredicted yield for F006: {predicted_yields[0]:.2f} tons/hectare")