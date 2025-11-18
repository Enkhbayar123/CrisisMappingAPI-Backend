from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# 1. The "Base" Class
# This contains fields that are common to both input and output.
# We do this to avoid typing 'text' and 'image_url' twice.
class EventBase(BaseModel):
    text: str
    image_url: str
    location_name: str  # Example: "Houston, TX"

# 2. The "Input" (Request)
# This is exactly what the Frontend sends to us.
class EventCreate(EventBase):
    pass 
    # No extra fields needed. It just looks like the Base.

# 3. The "Output" (Response)
# This is what we send back to the Frontend after processing.
class EventResponse(EventBase):
    id: int
    created_at: datetime
    
    # Geocoding Results
    latitude: float
    longitude: float

    # AI Results (Mandatory now, since we wait for them)
    severity: str        # "High", "Medium", "Low"
    category: str        # "Flood", "Fire", etc.
    is_informative: bool # True/False

    class Config:
        # This line is magic. It allows Pydantic to read data 
        # directly from your SQL Database models later.
        from_attributes = True