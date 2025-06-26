import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_yield_predictions(predictions_df):
    """
    Generates a bar plot of predicted yields per farm.
    predictions_df should have 'farm_id' and 'predicted_yield' columns.
    """
    if predictions_df.empty:
        print("No data to plot for yield predictions.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='farm_id', y='predicted_yield', data=predictions_df)
    plt.title('Predicted Agricultural Yields per Farm')
    plt.xlabel('Farm ID')
    plt.ylabel('Predicted Yield (tons/hectare)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots feature importance from a trained machine learning model.
    Assumes the model has a `feature_importances_` attribute (e.g., RandomForest).
    """
    if not hasattr(model, 'feature_importances_') or model is None:
        print("Model does not have feature importances or is not trained.")
        return

    importances = model.feature_importances_
    features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_df.head(10)) # Plot top 10 features
    plt.title('Top 10 Feature Importances for Yield Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def create_supply_chain_route_description(shipment_data_df):
    """
    Provides a textual description of shipment routes instead of a map
    (since Folium requires more setup and real coordinates for good visualization).
    """
    if shipment_data_df.empty:
        print("No shipment data to describe.")
        return

    print("\n--- Simulated Supply Chain Routes ---")
    for idx, row in shipment_data_df.iterrows():
        print(f"Shipment ID: {row['shipment_id']}")
        print(f"  From: {row['origin_loc']} (Lat: {row['origin_lat']:.4f}, Lon: {row['origin_lon']:.4f})")
        print(f"  To: {row['dest_loc']} (Lat: {row['dest_lat']:.4f}, Lon: {row['dest_lon']:.4f})")
        print(f"  Quantity: {row['quantity_kg']} kg")
        print(f"  Timestamp: {row['timestamp']}")
        print("-" * 30)

# Example usage (for internal testing)
if __name__ == "__main__":
    # Dummy data for plotting yield predictions
    predicted_yields_sample_df = pd.DataFrame({
        'farm_id': ['Farm_A', 'Farm_B', 'Farm_C', 'Farm_D'],
        'predicted_yield': [1.8, 2.1, 1.5, 2.3]
    })
    plot_yield_predictions(predicted_yields_sample_df)

    # Dummy model and feature names for feature importance
    class DummyModelWithImportance:
        def __init__(self, importances):
            self.feature_importances_ = np.array(importances)
    
    dummy_model_importances = [0.35, 0.25, 0.15, 0.1, 0.08, 0.04, 0.03]
    dummy_feature_names = [
        'avg_ndvi_growth_period', 'total_precip_growth_period', 
        'avg_temp_growth_period', 'soil_ph', 'previous_yield',
        'planting_month', 'soil_nitrogen'
    ]
    dummy_model = DummyModelWithImportance(dummy_model_importances)
    plot_feature_importance(dummy_model, dummy_feature_names)

    # Dummy data for supply chain routes (textual description)
    shipment_data_sample = pd.DataFrame({
        'shipment_id': ['S001', 'S002'],
        'origin_loc': ['Dodoma', 'Morogoro'],
        'origin_lat': [-6.1738, -6.8228],
        'origin_lon': [35.7479, 37.6698],
        'dest_loc': ['Dar es Salaam', 'Arusha'],
        'dest_lat': [-6.7924, -3.3869],
        'dest_lon': [39.2083, 36.6822],
        'quantity_kg': [1500, 1000],
        'timestamp': [datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(hours=24)]
    })
    create_supply_chain_route_description(shipment_data_sample)