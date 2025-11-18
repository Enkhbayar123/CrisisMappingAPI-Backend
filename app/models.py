from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from .database import Base

class CrisisEvent(Base):
    __tablename__ = "crisis_events"

    # 1. Metadata
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 2. User Inputs
    text = Column(Text)
    location_name = Column(String) # "Houston, TX"
    image_url = Column(String)     # "http://.../static/photo.jpg"

    # 3. Geolocation (The "PostGIS" magic)
    # We store Lat/Lon as simple floats for easy reading...
    latitude = Column(Float)
    longitude = Column(Float)
    # ...AND as a Geometry Point for fast map queries (e.g. "find within 5km")
    # SRID 4326 is the code for standard GPS Lat/Lon.
    geom = Column(Geometry(geometry_type='POINT', srid=4326))

    # 4. AI Results (The output of your research model)
    severity = Column(String)        # "High", "Medium", "Low"
    category = Column(String)        # "Flood", "Fire"
    is_informative = Column(Boolean) # True/False