import pandas as pd
import numpy as np
import datetime
from sqlalchemy.orm import Session
from database import FarmData # Import FarmData model

class DataProcessor:
    def __init__(self, data_config=None):
        self.config = data_config if data_config is not None else {}
        print("DataProcessor initialized (using simulated data sources).")

    def fetch_weather_data(self, lat, lon, start_date, end_date):
        """
        Simulates fetching historical weather data.
        """
        print(f"Simulating fetching weather data for {lat}, {lon} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
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
        """
        print(f"Simulating fetching {product} satellite data for {bounding_box} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='W') # Weekly NDVI for simplicity
        
        ndvi_values = 0.4 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, len(dates))) + np.random.rand(len(dates)) * 0.1
        ndvi_values = np.clip(ndvi_values, 0.2, 0.8)
        
        satellite_df = pd.DataFrame({
            'date': dates,
            'ndvi': ndvi_values
        })
        return satellite_df

    def load_soil_data(self, farm_ids: list, db_session: Session = None):
        """
        Simulates loading soil data for given farm IDs and persists/retrieves from DB.
        """
        print(f"Simulating loading soil data for farms: {farm_ids}...")
        
        soil_data_list = []
        for farm_id in farm_ids:
            # Try to fetch from DB first
            if db_session:
                farm_data_obj = db_session.query(FarmData).filter_by(farm_id=farm_id).first()
                if farm_data_obj:
                    soil_data_list.append({
                        'farm_id': farm_id,
                        'soil_type': farm_data_obj.soil_type,
                        'ph': farm_data_obj.ph,
                        'nitrogen': farm_data_obj.nitrogen,
                        'phosphorus': farm_data_obj.phosphorus
                    })
                    continue # Already have data for this farm
            
            # If not in DB or no session, generate synthetic data
            soil_data_list.append({
                'farm_id': farm_id,
                'soil_type': np.random.choice(['loam', 'clay', 'sandy']),
                'ph': np.round(np.random.uniform(6.0, 7.5), 1),
                'nitrogen': np.round(np.random.uniform(20, 50), 1),
                'phosphorus': np.round(np.random.uniform(10, 30), 1),
            })
        
        soil_df = pd.DataFrame(soil_data_list)
        return soil_df

    def load_farm_practices_data(self, farm_ids: list, db_session: Session = None):
        """
        Simulates loading data on agricultural practices and persists/retrieves from DB.
        Also creates/updates FarmData entry.
        """
        print(f"Simulating loading farm practices data for farms: {farm_ids}...")
        
        farm_practices_list = []
        for farm_id in farm_ids:
            # Check if farm_id exists in FarmData table
            farm_data_obj = None
            if db_session:
                farm_data_obj = db_session.query(FarmData).filter_by(farm_id=farm_id).first()

            if farm_data_obj:
                # Use existing data if found
                farm_practices_list.append({
                    'farm_id': farm_id,
                    'crop_type': farm_data_obj.crop_type,
                    'planting_date': farm_data_obj.planting_date.strftime('%Y-%m-%d') if farm_data_obj.planting_date else None,
                    'fertilizer_applied': farm_data_obj.fertilizer_applied,
                    'irrigation_method': farm_data_obj.irrigation_method,
                    'previous_yield': farm_data_obj.previous_yield
                })
            else:
                # Generate synthetic data if not found
                synthetic_planting_date = datetime.date(2024, 3, 1) + datetime.timedelta(days=int(np.random.rand() * 60))
                new_farm_practice = {
                    'farm_id': farm_id,
                    'crop_type': np.random.choice(['maize', 'beans', 'rice']),
                    'planting_date': synthetic_planting_date.strftime('%Y-%m-%d'),
                    'fertilizer_applied': np.random.choice([True, False]),
                    'irrigation_method': np.random.choice(['rainfed', 'drip', 'sprinkler']),
                    'previous_yield': np.round(np.random.uniform(1.0, 3.0), 2)
                }
                farm_practices_list.append(new_farm_practice)

                # If session is available, create a new FarmData entry
                if db_session:
                    new_farm_data_obj = FarmData(
                        farm_id=farm_id,
                        soil_type=np.random.choice(['loam', 'clay', 'sandy']), # Will be updated by load_soil_data if it runs later
                        ph=np.round(np.random.uniform(6.0, 7.5), 1),
                        nitrogen=np.round(np.random.uniform(20, 50), 1),
                        phosphorus=np.round(np.random.uniform(10, 30), 1),
                        crop_type=new_farm_practice['crop_type'],
                        planting_date=synthetic_planting_date,
                        fertilizer_applied=str(new_farm_practice['fertilizer_applied']), # Store as string for flexibility
                        irrigation_method=new_farm_practice['irrigation_method'],
                        previous_yield=new_farm_practice['previous_yield']
                    )
                    db_session.add(new_farm_data_obj)
                    db_session.commit()
                    db_session.refresh(new_farm_data_obj) # Refresh to get ID if needed
                    print(f"Created new FarmData entry for {farm_id} in DB.")

        return pd.DataFrame(farm_practices_list)

    def preprocess_data(self, weather_df, satellite_df, soil_df, farm_df, farm_id, current_date, db_session: Session = None):
        """
        Performs data cleaning, merging, and feature engineering for a specific farm.
        Also updates FarmData in DB if soil data changes.
        """
        print(f"Preprocessing and merging data for farm: {farm_id}...")
        
        farm_specific_soil = soil_df[soil_df['farm_id'] == farm_id].iloc[0] if not soil_df[soil_df['farm_id'] == farm_id].empty else pd.Series()
        farm_specific_practices = farm_df[farm_df['farm_id'] == farm_id].iloc[0] if not farm_df[farm_df['farm_id'] == farm_id].empty else pd.Series()

        # Update FarmData object in DB with soil info if available and session provided
        if db_session and not farm_specific_soil.empty:
            farm_data_obj = db_session.query(FarmData).filter_by(farm_id=farm_id).first()
            if farm_data_obj:
                farm_data_obj.soil_type = farm_specific_soil.get('soil_type', farm_data_obj.soil_type)
                farm_data_obj.ph = farm_specific_soil.get('ph', farm_data_obj.ph)
                farm_data_obj.nitrogen = farm_specific_soil.get('nitrogen', farm_data_obj.nitrogen)
                farm_data_obj.phosphorus = farm_specific_soil.get('phosphorus', farm_data_obj.phosphorus)
                db_session.add(farm_data_obj)
                db_session.commit()
                print(f"Updated FarmData entry for {farm_id} with soil info in DB.")

        weather_relevant = weather_df[weather_df['date'] <= current_date].tail(90)
        avg_temp_growth = weather_relevant['avg_temp'].mean()
        total_precip_growth = weather_relevant['total_precipitation'].sum()

        satellite_relevant = satellite_df[satellite_df['date'] <= current_date].tail(4)
        avg_ndvi_growth = satellite_relevant['ndvi'].mean()
        
        processed_data = {
            'farm_id': farm_id,
            'planting_month': pd.to_datetime(farm_specific_practices.get('planting_date')).month if not farm_specific_practices.empty and pd.notna(farm_specific_practices.get('planting_date')) else 0,
            'avg_temp_growth_period': avg_temp_growth if pd.notna(avg_temp_growth) else 25,
            'total_precip_growth_period': total_precip_growth if pd.notna(total_precip_growth) else 300,
            'avg_ndvi_growth_period': avg_ndvi_growth if pd.notna(avg_ndvi_growth) else 0.6,
            'soil_ph': farm_specific_soil.get('ph', 6.5),
            'soil_nitrogen': farm_specific_soil.get('nitrogen', 35),
            'previous_yield': farm_specific_practices.get('previous_yield', 1.5),
        }
        return pd.DataFrame([processed_data])

# Example usage (for internal testing) - Remains similar, but DB interaction will happen through Orchestrator
if __name__ == "__main__":
    processor = DataProcessor()
    
    test_farm_id = "TEST_FARM_001"
    test_lat, test_lon = -6.1738, 35.7479
    test_current_date = datetime.datetime.now()

    # Note: Direct DB interaction here would require creating a session manually
    # For simplicity, this example doesn't run with direct DB save,
    # as the Orchestrator handles passing the session.
    weather_data = processor.fetch_weather_data(test_lat, test_lon, test_current_date - datetime.timedelta(days=180), test_current_date)
    satellite_data = processor.fetch_satellite_data([test_lon-0.1, test_lat-0.1, test_lon+0.1, test_lat+0.1], test_current_date - datetime.timedelta(days=180), test_current_date)
    
    # These calls would ideally get a db_session from an orchestrator/main entry point
    # For test, we'll let them generate synthetic data
    soil_data = processor.load_soil_data([test_farm_id])
    farm_practices_data = processor.load_farm_practices_data([test_farm_id])

    processed_df = processor.preprocess_data(weather_data, satellite_data, soil_data, farm_practices_data, test_farm_id, test_current_date)
    print("\nProcessed Data Sample (for ML input):")
    print(processed_df.head())