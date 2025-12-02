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
        category=ai_results["category"],
        
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