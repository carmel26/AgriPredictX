# dashboard_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
import uuid
import os

# Import database functions and models
from database import get_db, FarmData, Prediction, ActualYield, Shipment
# Import the orchestrator directly
from main_system_orchestrator import AgriPredictXOrchestrator

# Configure logging for better visibility in Streamlit's console
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


st.set_page_config(layout="wide")
st.title("🌱 AgriPredict-X Dashboard for Tanzania 🇹🇿")

st.markdown("""
This dashboard provides insights into agricultural yield predictions and supply chain movements.
Data displayed here is sourced from the SQLite database populated by the AgriPredict-X system.
""")

# --- Data Fetching from Database (with cache) ---
@st.cache_data # Cache data for performance
def fetch_prediction_data_from_db():
    logging.info("Fetching prediction data from DB (or cache)...")
    with get_db() as db:
        predictions = db.query(Prediction).all()
        actuals = db.query(ActualYield).all()
        farm_data_records = db.query(FarmData).all()
    
    predictions_data = []
    for p in predictions:
        predictions_data.append({
            'id': p.id,
            'farm_id': p.farm_id,
            'predicted_yield': p.predicted_yield,
            'timestamp': p.timestamp
        })
    predictions_df = pd.DataFrame(predictions_data) if predictions_data else pd.DataFrame()

    actuals_data = []
    for a in actuals:
        actuals_data.append({
            'id': a.id,
            'farm_id': a.farm_id,
            'actual_yield': a.actual_yield,
            'timestamp': a.timestamp
        })
    actuals_df = pd.DataFrame(actuals_data) if actuals_data else pd.DataFrame()

    farm_data_processed = []
    for f in farm_data_records:
        farm_data_processed.append({
            'farm_id': f.farm_id,
            'crop_type': f.crop_type,
            'soil_type': f.soil_type,
            'ph': f.ph,
            'nitrogen': f.nitrogen,
            'phosphorus': f.phosphorus,
            'planting_date': f.planting_date,
            'fertilizer_applied': f.fertilizer_applied,
            'irrigation_method': f.irrigation_method,
            'previous_yield': f.previous_yield
        })
    farm_data_df = pd.DataFrame(farm_data_processed) if farm_data_processed else pd.DataFrame()


    # Merge predictions and actuals
    if not predictions_df.empty:
        predictions_df = predictions_df.rename(columns={'timestamp': 'predicted_timestamp'})
        predictions_df = predictions_df.set_index('farm_id')
        
        if not actuals_df.empty:
            actuals_df = actuals_df.rename(columns={'timestamp': 'actual_timestamp'})
            actuals_df = actuals_df.set_index('farm_id')
            combined_df = predictions_df.merge(actuals_df, on='farm_id', how='left', suffixes=('_pred', '_actual'))
        else:
            combined_df = predictions_df.copy()
            combined_df['actual_yield'] = np.nan
            combined_df['actual_timestamp'] = pd.NaT
        combined_df = combined_df.reset_index()
        
        # Add crop_type from farm_data_df
        if not farm_data_df.empty:
            combined_df = combined_df.merge(farm_data_df[['farm_id', 'crop_type']], on='farm_id', how='left')
        
        # Add a dummy region for demonstration if not in DB (or if farm_data_df is empty)
        if 'region' not in combined_df.columns:
            def assign_region(farm_id):
                if farm_id.startswith("Dodoma"): return "Dodoma"
                if farm_id.startswith("Mbeya"): return "Mbeya"
                if farm_id.startswith("Morogoro"): return "Morogoro"
                if farm_id.startswith("Arusha"): return "Arusha"
                if farm_id.startswith("Iringa"): return "Iringa"
                if farm_id.startswith("Tabora"): return "Tabora"
                return np.random.choice(['Dodoma', 'Mbeya', 'Morogoro', 'Arusha', 'Iringa', 'Tabora'])

            combined_df['region'] = combined_df['farm_id'].apply(assign_region)
            
    else:
        combined_df = pd.DataFrame(columns=['farm_id', 'predicted_yield', 'actual_yield', 'predicted_timestamp', 'actual_timestamp', 'region', 'crop_type'])

    logging.info(f"Fetched prediction data. Contains {len(combined_df['farm_id'].unique()) if not combined_df.empty else 0} unique farms.")
    return combined_df

@st.cache_data
def fetch_shipment_data_from_db():
    logging.info("Fetching shipment data from DB (or cache)...")
    with get_db() as db:
        shipments = db.query(Shipment).all()
    
    shipments_data = []
    for s in shipments:
        shipments_data.append({
            'id': s.id,
            'shipment_id': s.shipment_id,
            'farm_id': s.farm_id,
            'quantity_kg': s.quantity_kg,
            'origin_loc': s.origin_loc,
            'dest_loc': s.dest_loc,
            'timestamp': s.timestamp
        })
    shipments_df = pd.DataFrame(shipments_data) if shipments_data else pd.DataFrame()
    logging.info(f"Fetched shipment data. Contains {len(shipments_df)} shipments.")
    return shipments_df


# --- Initialize and Cache Orchestrator ---
@st.cache_resource
def get_orchestrator():
    orchestrator = AgriPredictXOrchestrator()
    orchestrator.initial_setup_and_training()
    
    with get_db() as db:
        first_prediction = db.query(Prediction).first()
    
    if not first_prediction:
        logging.info("No existing prediction data found. Populating initial dummy data.")
        orchestrator.populate_initial_dummy_data()
        fetch_prediction_data_from_db.clear()
        fetch_shipment_data_from_db.clear()
    
    return orchestrator

orchestrator = get_orchestrator()

# --- Functions to trigger actions ---
def trigger_new_prediction_cycle():
    fetch_prediction_data_from_db.clear()
    fetch_shipment_data_from_db.clear()
    
    with get_db() as db:
        current_farm_count = db.query(FarmData).count()
    new_farm_id = f"NewFarm_{current_farm_count + 1:03d}"
    
    new_lat = np.random.uniform(-11.5, -1.0)
    new_lon = np.random.uniform(29.5, 40.5)
    current_time_for_new_data = datetime.datetime.now() + datetime.timedelta(minutes=np.random.randint(1, 60))
    
    with st.spinner(f"Running prediction cycle for {new_farm_id}..."):
        orchestrator.run_prediction_cycle(
            farm_id=new_farm_id,
            lat=new_lat,
            lon=new_lon,
            current_date=current_time_for_new_data
        )
        orchestrator.run_supply_chain_logging_event(
            event_type="actual_yield",
            farm_id=new_farm_id,
            actual_yield=round(np.random.uniform(1.0, 3.5), 2),
            timestamp=current_time_for_new_data + datetime.timedelta(days=np.random.randint(20, 40))
        )
        orchestrator.run_supply_chain_logging_event(
            event_type="shipment",
            shipment_id=str(uuid.uuid4()),
            farm_id=new_farm_id,
            quantity_kg=np.random.randint(800, 2500),
            origin_loc=f"{new_farm_id.split('_')[0] if '_' in new_farm_id else 'Generated'} Local",
            dest_loc=np.random.choice(['Dar es Salaam Central Market', 'Arusha Distribution', 'Zanzibar Port']),
            timestamp=current_time_for_new_data + datetime.timedelta(days=np.random.randint(45, 70))
        )
    st.success(f"New prediction cycle for {new_farm_id} completed and data saved!")
    st.rerun()

def retrain_ml_model():
    get_orchestrator.clear() 
    fetch_prediction_data_from_db.clear()
    fetch_shipment_data_from_db.clear()
    
    with st.spinner("Retraining ML Model... This might take a moment."):
        orchestrator = get_orchestrator()
    st.success("ML Model retraining completed!")
    st.rerun()

def clear_database_content_and_repopulate():
    """
    Deletes all records from the database tables but keeps the table structure.
    Then repopulates with initial dummy data.
    """
    with st.spinner("Clearing database content and repopulating..."):
        with get_db() as db:
            # Delete all records from each table
            db.query(Shipment).delete()
            db.query(ActualYield).delete()
            db.query(Prediction).delete()
            db.query(FarmData).delete()
            db.commit()
            logging.info("All existing database records deleted.")
        
        # Clear Streamlit caches so new data is fetched
        fetch_prediction_data_from_db.clear()
        fetch_shipment_data_from_db.clear()
        
        # Repopulate with dummy data
        orchestrator.populate_initial_dummy_data()
        
    st.success("Database content cleared and re-populated with fresh dummy data (structure preserved)!")
    st.rerun()


# --- Dashboard Layout ---
predictions_actuals_df = fetch_prediction_data_from_db()
shipments_df = fetch_shipment_data_from_db()

st.header("Yield Prediction Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Farms Monitored", len(predictions_actuals_df['farm_id'].unique()) if not predictions_actuals_df.empty else 0)
with col2:
    st.metric("Avg Predicted Yield (t/ha)", f"{predictions_actuals_df['predicted_yield'].mean():.2f}" if not predictions_actuals_df.empty else "N/A")
with col3:
    st.metric("Total Shipments Recorded", len(shipments_df) if not shipments_df.empty else 0)

if not predictions_actuals_df.empty:
    st.subheader("Predicted vs. Actual Yields by Farm")
    selected_region = st.selectbox("Filter by Region", ['All'] + sorted(list(predictions_actuals_df['region'].unique())))
    if selected_region != 'All':
        filtered_predictions = predictions_actuals_df[predictions_actuals_df['region'] == selected_region]
    else:
        filtered_predictions = predictions_actuals_df

    if not filtered_predictions.empty:
        fig_yield_comp, ax_yield_comp = plt.subplots(figsize=(12, 6))
        plot_df = filtered_predictions[['farm_id', 'predicted_yield', 'actual_yield']].set_index('farm_id')
        plot_df.plot(kind='bar', ax=ax_yield_comp, width=0.8)
        ax_yield_comp.set_ylabel("Yield (tons/hectare)")
        ax_yield_comp.set_title(f"Predicted vs. Actual Yields in {selected_region}")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_yield_comp)
    else:
        st.info("No prediction data for the selected region or farms in this region.")

    st.subheader("Yield Trends Over Time")
    predictions_actuals_df['predicted_timestamp'] = pd.to_datetime(predictions_actuals_df['predicted_timestamp'])
    predictions_actuals_df_sorted = predictions_actuals_df.sort_values('predicted_timestamp')

    fig_yield_trend, ax_yield_trend = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='predicted_timestamp', y='predicted_yield', data=predictions_actuals_df_sorted, label='Predicted', ax=ax_yield_trend)
    if 'actual_yield' in predictions_actuals_df_sorted.columns and not predictions_actuals_df_sorted['actual_yield'].isnull().all():
        sns.lineplot(x='predicted_timestamp', y='actual_yield', data=predictions_actuals_df_sorted.dropna(subset=['actual_yield']), label='Actual', ax=ax_yield_trend)
    ax_yield_trend.set_xlabel("Date")
    ax_yield_trend.set_ylabel("Yield (tons/hectare)")
    ax_yield_trend.set_title("Overall Yield Trend")
    ax_yield_trend.tick_params(axis='x', rotation=45)
    ax_yield_trend.legend()
    st.pyplot(fig_yield_trend)
else:
    st.info("No yield prediction data available yet. Use the sidebar buttons to generate data.")

st.header("Supply Chain Monitoring")
if not shipments_df.empty:
    st.subheader("Shipment Volume by Destination")
    fig_shipment_dest, ax_shipment_dest = plt.subplots(figsize=(10, 5))
    shipments_df.groupby('dest_loc')['quantity_kg'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax_shipment_dest)
    ax_shipment_dest.set_ylabel("Total Quantity (kg)")
    ax_shipment_dest.set_xlabel("Destination Location")
    ax_shipment_dest.set_title("Total Shipment Volume by Destination")
    st.pyplot(fig_shipment_dest)

    st.subheader("Recent Shipments Log")
    st.dataframe(shipments_df.sort_values('timestamp', ascending=False).head(15))
else:
    st.info("No shipment data available yet. Use the sidebar buttons to generate data.")

st.sidebar.header("AgriPredict-X Control")
st.sidebar.markdown("Use these buttons to interact with the system:")
if st.sidebar.button("Trigger New Prediction Cycle", use_container_width=True):
    trigger_new_prediction_cycle()
if st.sidebar.button("Retrain ML Model", use_container_width=True):
    retrain_ml_model()

st.sidebar.markdown("---")
st.sidebar.subheader("Database Actions:")
if st.sidebar.button("Clear Data (Keep Structure)", help="Deletes all data records from tables, then repopulates dummy data (keeps table structure).", use_container_width=True):
    clear_database_content_and_repopulate()

if st.sidebar.button("Clear All Caches & Reset Database File", help="Deletes the database file, clears all caches, and forces app to re-initialize with fresh dummy data.", use_container_width=True):
    if os.path.exists("agripredictx.db"):
        os.remove("agripredictx.db")
        st.info("Deleted existing database file.")
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("All caches cleared and database reset! App will re-initialize with fresh data.")
    st.rerun()