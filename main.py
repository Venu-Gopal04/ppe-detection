import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from detector import detect_ppe_in_image, detect_ppe_in_video

load_dotenv()
# Create required directories on startup
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("frames", exist_ok=True)
app = FastAPI(title="PPE Detection System")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Upload an image for PPE detection."""
    allowed = [".jpg", ".jpeg", ".png", ".bmp"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only image files supported (JPG, PNG).")

    # Save uploaded file
    upload_path = f"uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run detection
    output_path = f"outputs/detected_{file.filename}"
    result = detect_ppe_in_image(upload_path, output_path)

    return {
        "message": "Detection complete!",
        "person_count": result["person_count"],
        "total_detections": result["total_detections"],
        "safety_status": result["safety_status"],
        "output_image_url": f"/outputs/detected_{file.filename}",
        "detections": result["detections"]
    }

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """Upload a video for PPE detection."""
    allowed = [".mp4", ".avi", ".mov", ".mkv"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only video files supported.")

    upload_path = f"uploads/{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_path = f"outputs/detected_{file.filename}"
    result = detect_ppe_in_video(upload_path, output_path, frame_interval=15)

    return {
        "message": "Video detection complete!",
        "total_frames": result["total_frames"],
        "frames_with_persons": result["frames_with_persons"],
        "max_persons_in_frame": result["max_persons_in_frame"],
        "safety_summary": result["safety_summary"],
        "alert_timestamps": result["alert_timestamps"],
        "output_video_url": f"/outputs/detected_{file.filename}"
    }

@app.get("/health")
def health():
    return {"status": "PPE Detection System is running!"}