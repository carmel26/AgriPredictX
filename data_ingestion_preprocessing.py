import pandas as pd
import numpy as np
import datetime

class DataProcessor:
    def __init__(self, data_config=None):
        self.config = data_config if data_config is not None else {}
        print("DataProcessor initialized (using simulated data sources).")

    def fetch_weather_data(self, lat, lon, start_date, end_date):
        """
        Simulates fetching historical weather data.
        In a real system, this would connect to a weather API (e.g., OpenWeatherMap).
        """
        print(f"Simulating fetching weather data for {lat}, {lon} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate temperature and precipitation variations
        temperatures = 25 + 5 * np.sin(np.linspace(0, 2 * np.pi, len(dates))) + np.random.rand(len(dates)) * 2
        precipitations = np.random.rand(len(dates)) * 10 # mm
        
        weather_df = pd.DataFrame({
            'date': dates,
            'avg_temp': temperatures,
            'total_precipitation': precipitations
        })
        return weather_df

    def fetch_satellite_data(self, bounding_box, start_date, end_date, product='NDVI'):
        """
        Simulates fetching satellite imagery derived products (e.g., NDVI).
        In a real system, this would connect to services like Google Earth Engine or Sentinel Hub.
        """
        print(f"Simulating fetching {product} satellite data for {bounding_box} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='W') # Weekly NDVI for simplicity
        
        # Simulate NDVI values (0.2 for bare soil to 0.8 for healthy vegetation)
        ndvi_values = 0.4 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, len(dates))) + np.random.rand(len(dates)) * 0.1
        ndvi_values = np.clip(ndvi_values, 0.2, 0.8) # Keep within typical NDVI range
        
        satellite_df = pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values
        })
        return satellite_df

    def load_soil_data(self, farm_ids):
        """
        Simulates loading soil data for given farm IDs.
        In a real system, this might load from a database or file.
        """
        print(f"Simulating loading soil data for farms: {farm_ids}...")
        
        # Create synthetic soil data for demonstration
        soil_data = {
            'farm_id': farm_ids,
            'soil_type': np.random.choice(['loam', 'clay', 'sandy'], len(farm_ids)),
            'ph': np.round(np.random.uniform(6.0, 7.5, len(farm_ids)), 1),
            'nitrogen': np.round(np.random.uniform(20, 50, len(farm_ids)), 1), # kg/ha
            'phosphorus': np.round(np.random.uniform(10, 30, len(farm_ids)), 1), # kg/ha
        }
        return pd.DataFrame(soil_data)

    def load_farm_practices_data(self, farm_ids):
        """
        Simulates loading data on agricultural practices.
        In a real system, this could come from farmer inputs via a mobile app.
        """
        print(f"Simulating loading farm practices data for farms: {farm_ids}...")
        
        farm_practices = {
            'farm_id': farm_ids,
            'crop_type': np.random.choice(['maize', 'beans', 'rice'], len(farm_ids)),
            'planting_date': [
                (datetime.date(2024, 3, 1) + datetime.timedelta(days=int(np.random.rand() * 60))).strftime('%Y-%m-%d')
                for _ in farm_ids
            ],
            'fertilizer_applied': np.random.choice([True, False], len(farm_ids)),
            'irrigation_method': np.random.choice(['rainfed', 'drip', 'sprinkler'], len(farm_ids)),
            'previous_yield': np.round(np.random.uniform(1.0, 3.0, len(farm_ids)), 2) # tons/hectare
        }
        return pd.DataFrame(farm_practices)

    def preprocess_data(self, weather_df, satellite_df, soil_df, farm_df, farm_id, current_date):
        """
        Performs data cleaning, merging, and feature engineering for a specific farm.
        This is a simplified example; real feature engineering would be more complex.
        """
        print(f"Preprocessing and merging data for farm: {farm_id}...")
        
        # Filter farm-specific data
        farm_specific_soil = soil_df[soil_df['farm_id'] == farm_id].iloc[0] if not soil_df[soil_df['farm_id'] == farm_id].empty else pd.Series()
        farm_specific_practices = farm_df[farm_df['farm_id'] == farm_id].iloc[0] if not farm_df[farm_df['farm_id'] == farm_id].empty else pd.Series()

        # Aggregate weather data for a relevant period (e.g., last 3 months up to current date)
        weather_relevant = weather_df[weather_df['date'] <= current_date].tail(90) # Last 90 days
        avg_temp_growth = weather_relevant['avg_temp'].mean()
        total_precip_growth = weather_relevant['total_precipitation'].sum()

        # Aggregate satellite data for a relevant period
        satellite_relevant = satellite_df[satellite_df['date'] <= current_date].tail(4) # Last 4 weeks
        avg_ndvi_growth = satellite_relevant['ndvi'].mean()
        
        # Create a single row DataFrame for the current farm's features
        processed_data = {
            'farm_id': farm_id,
            'planting_month': pd.to_datetime(farm_specific_practices.get('planting_date')).month if not farm_specific_practices.empty else 0,
            'avg_temp_growth_period': avg_temp_growth if not pd.isna(avg_temp_growth) else 25,
            'total_precip_growth_period': total_precip_growth if not pd.isna(total_precip_growth) else 300,
            'avg_ndvi_growth_period': avg_ndvi_growth if not pd.isna(avg_ndvi_growth) else 0.6,
            'soil_ph': farm_specific_soil.get('ph', 6.5),
            'soil_nitrogen': farm_specific_soil.get('nitrogen', 35),
            'previous_yield': farm_specific_practices.get('previous_yield', 1.5),
            # Add more features as needed
        }
        return pd.DataFrame([processed_data])

# Example usage (for internal testing)
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Define a dummy farm and date for testing
    test_farm_id = "TEST_FARM_001"
    test_lat, test_lon = -6.1738, 35.7479 # Dodoma, Tanzania
    test_current_date = datetime.datetime.now()

    # Simulate fetching/loading data
    weather_data = processor.fetch_weather_data(test_lat, test_lon, test_current_date - datetime.timedelta(days=180), test_current_date)
    satellite_data = processor.fetch_satellite_data([test_lon-0.1, test_lat-0.1, test_lon+0.1, test_lat+0.1], test_current_date - datetime.timedelta(days=180), test_current_date)
    soil_data = processor.load_soil_data([test_farm_id])
    farm_practices_data = processor.load_farm_practices_data([test_farm_id])

    # Preprocess
    processed_df = processor.preprocess_data(weather_data, satellite_data, soil_data, farm_practices_data, test_farm_id, test_current_date)
    print("\nProcessed Data Sample (for ML input):")
    print(processed_df.head())