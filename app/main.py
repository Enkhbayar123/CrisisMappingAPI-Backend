from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import your modules
from . import models, schemas, database
from .ai_core.service import ai_engine  # Import the global instance

from typing import Optional
from datetime import datetime
from fastapi import Query
from sqlalchemy import or_, func, cast
from geoalchemy2 import Geography


# --- LIFESPAN (The Startup Manager) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. ENSURE STATIC DIRECTORIES EXIST ON STARTUP
    os.makedirs("app/static/images", exist_ok=True)
    print("Checked/Created static directories.")

    # 1. Startup: Load the AI Model onto the GPU
    ai_engine.load_model()
    yield
    # 2. Shutdown: Clean up resources
    print("Server shutting down...")

# Initialize App with Lifespan
app = FastAPI(lifespan=lifespan)

# Define Allowed Origins
origins = [
        'http://localhost:3000',
        'https://crisis-mapping-frontend.vercel.app',
        'http://192.168.1.47:3000',
        'http://203.252.106.25:8000', 
      ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Ensure the static directory exists before mounting
os.makedirs("app/static", exist_ok=True)

# Mount Static Folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create Tables
models.Base.metadata.create_all(bind=database.engine)

# DB Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/events", response_model=list[schemas.EventResponse])
def get_events(db: Session = Depends(get_db)):
    return db.query(models.CrisisEvent).filter(models.CrisisEvent.is_informative == True).all()

# --- THE ENDPOINT ---
@app.post("/events", response_model=schemas.EventResponse)
async def create_event(
    file: UploadFile = File(...), 
    text: str = Form(...), 
    location_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    type: str = Form("Unknown"),
    db: Session = Depends(get_db)
):
    # 1. Save File
    upload_dir = "app/static/images"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_location = f"{upload_dir}/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    image_url = f"/static/images/{file.filename}"

    # 2. Run AI (Uses the pre-loaded GPU model)
    ai_results = ai_engine.predict(text, file_location)

    # 3. Save to DB
    new_event = models.CrisisEvent(
        text=text,
        location_name=location_name,
        image_url=image_url,
        latitude=latitude,
        longitude=longitude,
        geom=f"POINT({longitude} {latitude})", 
        
        # Unpack AI results
        severity=ai_results["severity"],
        humanitarian=ai_results["humanitarian"],
        type = type,
        
        # FIX: Convert the String "informative" to a Boolean True/False
        is_informative=(ai_results["is_informative"] == "informative")
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)


    if not new_event.is_informative:
        # If not informative, return a generic message.
        # This prevents the frontend from receiving the event details.
        return JSONResponse(
            status_code=200, 
            content={"message": "Report saved to database, but marked as not informative. It will not be displayed."}
        )
    
    return new_event

@app.get("/events", response_model=list[schemas.EventResponse])
def get_events(
    db: Session = Depends(get_db),
    # 1. Text Search (Keyword matching)
    search: Optional[str] = Query(None, description="Search in event text or location"),
    
    # 2. Categorical Filters
    type: Optional[str] = Query(None, description="Filter by disaster type"),             # NEW
    humanitarian: Optional[str] = Query(None, description="Filter by humanitarian need"), # NEW
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    
    # 3. Date Filters
    start_date: Optional[datetime] = Query(None, description="Events from this date"),
    end_date: Optional[datetime] = Query(None, description="Events until this date"),
    
    # 4. Geospatial Filters (PostGIS Magic)
    lat: Optional[float] = Query(None, description="Center latitude for radius search"),
    lon: Optional[float] = Query(None, description="Center longitude for radius search"),
    radius_km: Optional[float] = Query(None, description="Radius in kilometers"),
    
    # 5. Pagination (Crucial for performance)
    limit: int = Query(50, ge=1, le=100, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Records to skip")
):
    # Start with the base query: Only get informative events
    query = db.query(models.CrisisEvent).filter(models.CrisisEvent.is_informative == True)

    # Apply Text Search using ILIKE (case-insensitive)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                models.CrisisEvent.text.ilike(search_term),
                models.CrisisEvent.location_name.ilike(search_term)
            )
        )

    # Apply Exact Match Filters
    if type:
        query = query.filter(models.CrisisEvent.type == type)
    if humanitarian:
        query = query.filter(models.CrisisEvent.humanitarian == humanitarian)
    if severity:
        query = query.filter(models.CrisisEvent.severity == severity)

    # Apply Date Range Filters
    if start_date:
        query = query.filter(models.CrisisEvent.created_at >= start_date)
    if end_date:
        query = query.filter(models.CrisisEvent.created_at <= end_date)

    # Apply Geospatial Radius Search (Requires PostGIS)
    if lat is not None and lon is not None and radius_km is not None:
        # Convert radius from kilometers to meters
        radius_meters = radius_km * 1000
        
        # Create a WKT (Well-Known Text) point for the search center
        center_point = f"SRID=4326;POINT({lon} {lat})"
        
        # ST_DWithin calculates if geometries are within a given distance.
        # We cast the Geometry to Geography so PostgreSQL calculates distance 
        # accurately in meters over the Earth's curvature, not in flat map degrees.
        query = query.filter(
            func.ST_DWithin(
                cast(models.CrisisEvent.geom, Geography),
                cast(center_point, Geography),
                radius_meters
            )
        )

    # Always order by newest first, then apply pagination
    query = query.order_by(models.CrisisEvent.created_at.desc())
    events = query.offset(offset).limit(limit).all()

    return events