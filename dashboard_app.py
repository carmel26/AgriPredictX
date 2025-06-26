# dashboard_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sqlalchemy.orm import sessionmaker # If using SQLAlchemy
# from db_models import Prediction, Shipment, engine # Your database models and engine

# Placeholder for database session (replace with actual DB connection)
def get_db_session():
    # Session = sessionmaker(bind=engine)
    # return Session()
    # For demo, return None and use dummy data
    return None

st.set_page_config(layout="wide")
st.title("🌱 AgriPredict-X Dashboard for Tanzania 🇹🇿")

st.markdown("""
This dashboard provides insights into agricultural yield predictions and supply chain movements.
Data displayed here would ideally be pulled from a persistent database and verified on a blockchain.
""")

# --- Data Fetching (Mocked for Demo) ---
@st.cache_data # Cache data for performance
def fetch_mock_prediction_data():
    # session = get_db_session()
    # if session:
    #     predictions = session.query(Prediction).all()
    #     return pd.DataFrame([p.__dict__ for p in predictions])
    # Mock data for demonstration
    return pd.DataFrame({
        'farm_id': [f'Farm_{i:02d}' for i in range(1, 11)],
        'predicted_yield': np.random.uniform(1.5, 3.5, 10),
        'actual_yield': np.random.uniform(1.4, 3.6, 10),
        'timestamp': pd.to_datetime(['2025-06-01', '2025-06-05', '2025-06-10', '2025-06-15', '2025-06-20',
                                     '2025-06-02', '2025-06-07', '2025-06-12', '2025-06-17', '2025-06-22']),
        'region': np.random.choice(['Dodoma', 'Mbeya', 'Morogoro'], 10)
    })

@st.cache_data
def fetch_mock_shipment_data():
    # session = get_db_session()
    # if session:
    #     shipments = session.query(Shipment).all()
    #     return pd.DataFrame([s.__dict__ for s in shipments])
    # Mock data for demonstration
    return pd.DataFrame({
        'shipment_id': [f'S{i:03d}' for i in range(1, 6)],
        'farm_id': np.random.choice([f'Farm_{i:02d}' for i in range(1, 11)], 5),
        'quantity_kg': np.random.randint(500, 5000, 5),
        'origin_loc': np.random.choice(['Dodoma', 'Mbeya', 'Morogoro'], 5),
        'dest_loc': np.random.choice(['Dar es Salaam', 'Arusha', 'Mwanza'], 5),
        'timestamp': pd.to_datetime(['2025-06-03', '2025-06-08', '2025-06-13', '2025-06-18', '2025-06-23'])
    })

predictions_df = fetch_mock_prediction_data()
shipments_df = fetch_mock_shipment_data()

# --- Dashboard Layout ---
st.header("Yield Prediction Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Farms Monitored", len(predictions_df['farm_id'].unique()))
with col2:
    st.metric("Avg Predicted Yield (t/ha)", f"{predictions_df['predicted_yield'].mean():.2f}")
with col3:
    st.metric("Total Shipments Recorded", len(shipments_df))

st.subheader("Predicted vs. Actual Yields by Farm")
# Filter by region
selected_region = st.selectbox("Filter by Region", ['All'] + list(predictions_df['region'].unique()))
if selected_region != 'All':
    filtered_predictions = predictions_df[predictions_df['region'] == selected_region]
else:
    filtered_predictions = predictions_df

fig_yield_comp, ax_yield_comp = plt.subplots(figsize=(12, 6))
filtered_predictions[['predicted_yield', 'actual_yield']].plot(kind='bar', ax=ax_yield_comp, width=0.8)
ax_yield_comp.set_xticks(range(len(filtered_predictions)))
ax_yield_comp.set_xticklabels(filtered_predictions['farm_id'], rotation=45, ha='right')
ax_yield_comp.set_ylabel("Yield (tons/hectare)")
ax_yield_comp.set_title("Predicted vs. Actual Yields")
st.pyplot(fig_yield_comp)

st.subheader("Yield Trends Over Time")
# Ensure timestamp is datetime type for sorting
predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
predictions_df_sorted = predictions_df.sort_values('timestamp')

fig_yield_trend, ax_yield_trend = plt.subplots(figsize=(12, 6))
sns.lineplot(x='timestamp', y='predicted_yield', data=predictions_df_sorted, label='Predicted', ax=ax_yield_trend)
sns.lineplot(x='timestamp', y='actual_yield', data=predictions_df_sorted, label='Actual', ax=ax_yield_trend)
ax_yield_trend.set_xlabel("Date")
ax_yield_trend.set_ylabel("Yield (tons/hectare)")
ax_yield_trend.set_title("Overall Yield Trend")
ax_yield_trend.tick_params(axis='x', rotation=45)
ax_yield_trend.legend()
st.pyplot(fig_yield_trend)


st.header("Supply Chain Monitoring")
st.subheader("Shipment Volume by Destination")
fig_shipment_dest, ax_shipment_dest = plt.subplots(figsize=(10, 5))
shipments_df.groupby('dest_loc')['quantity_kg'].sum().sort_values(ascending=False).plot(kind='bar', ax=ax_shipment_dest)
ax_shipment_dest.set_ylabel("Total Quantity (kg)")
ax_shipment_dest.set_xlabel("Destination Location")
ax_shipment_dest.set_title("Total Shipment Volume by Destination")
st.pyplot(fig_shipment_dest)

st.subheader("Recent Shipments Log")
st.dataframe(shipments_df.sort_values('timestamp', ascending=False).head(10))

st.sidebar.header("AgriPredict-X Control (Future Integration)")
st.sidebar.button("Run New Prediction Cycle") # Dummy button
st.sidebar.button("Retrain ML Model") # Dummy button