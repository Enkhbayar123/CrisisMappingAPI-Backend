from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# 1. The Base (Shared Data)
class EventBase(BaseModel):
    text: str
    location_name: str

class EventCreate(EventBase):
    pass

# 2. The Output (Response)
# This is what we return to the user.
# We return a URL so the frontend can display the image we just saved.
class EventResponse(EventBase):
    id: int
    image_url: str          # We generate this path
    created_at: datetime
    
    # Geocoding Results
    latitude: float
    longitude: float

    # AI Results
    severity: str        
    category: str        
    is_informative: bool 

    class Config:
        from_attributes = True

# Note: We DO NOT define an "EventInput" schema here.
# Why? Because in FastAPI, file uploads are handled directly in the 
# route function arguments, not via Pydantic schemas.