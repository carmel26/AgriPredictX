# database.py

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import os
from contextlib import contextmanager # <--- ADD THIS IMPORT

# Define the base class for declarative models
Base = declarative_base()

# Define your database models (tables)
class FarmData(Base):
    __tablename__ = 'farm_data'
    id = Column(Integer, primary_key=True)
    farm_id = Column(String, unique=True, index=True)
    soil_type = Column(String)
    ph = Column(Float)
    nitrogen = Column(Float)
    phosphorus = Column(Float)
    crop_type = Column(String)
    planting_date = Column(DateTime)
    fertilizer_applied = Column(String) # or Boolean
    irrigation_method = Column(String)
    previous_yield = Column(Float)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    farm_id = Column(String, ForeignKey('farm_data.farm_id'), index=True)
    predicted_yield = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    # Potentially add references to weather_data_id, satellite_data_id for full traceability

class ActualYield(Base):
    __tablename__ = 'actual_yields'
    id = Column(Integer, primary_key=True)
    farm_id = Column(String, ForeignKey('farm_data.farm_id'), index=True)
    actual_yield = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    linked_prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=True) # Link to a prediction

class Shipment(Base):
    __tablename__ = 'shipments'
    id = Column(Integer, primary_key=True)
    shipment_id = Column(String, unique=True, index=True)
    farm_id = Column(String, ForeignKey('farm_data.farm_id'), index=True)
    quantity_kg = Column(Float)
    origin_loc = Column(String)
    dest_loc = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now)

# Database setup
DATABASE_URL = "sqlite:///agripredictx.db" # Using SQLite for simplicity

# Create a database engine
engine = create_engine(DATABASE_URL)

# Create a sessionmaker to create new session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_db_tables():
    """Creates all defined tables in the database."""
    Base.metadata.create_all(bind=engine)
    print(f"Database tables created or already exist at {DATABASE_URL}")

@contextmanager # <--- ADD THIS DECORATOR
def get_db():
    """Dependency for getting a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# You can call create_db_tables() once, e.g., from your orchestrator's initial setup.
if __name__ == "__main__":
    create_db_tables()
    print("Database setup script executed.")