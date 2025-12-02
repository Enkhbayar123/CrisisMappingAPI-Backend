from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware

# Import your modules
from . import models, schemas, database
from .ai_core.service import ai_engine  # Import the global instance

# --- LIFESPAN (The Startup Manager) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load the AI Model onto the GPU
    ai_engine.load_model()
    yield
    # 2. Shutdown: Clean up resources (if needed)
    print("Server shutting down...")

# Initialize App with Lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
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
    # Fetch all crisis events from the database
    return db.query(models.CrisisEvent).all()
# --- THE ENDPOINT ---
@app.post("/events", response_model=schemas.EventResponse)
async def create_event(
    file: UploadFile = File(...), 
    text: str = Form(...), 
    location_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    db: Session = Depends(get_db)
):
    # 1. Save File
    file_location = f"app/static/images/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image_url = f"/static/images/{file.filename}"

    # 2. Run AI (Uses the pre-loaded GPU model)
    # This runs on your LOCAL SERVER resources
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
        category=ai_results["category"],
        is_informative=ai_results["is_informative"]
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return new_event
