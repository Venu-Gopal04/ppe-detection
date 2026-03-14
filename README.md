# 🦺 PPE Detection System

AI-powered workplace safety monitoring using YOLOv8 computer vision.

## 🚀 What it does
- Upload any image or video
- YOLOv8s automatically detects persons in every frame
- Draws bounding boxes around detected persons
- Generates alert timestamps for every frame with persons detected
- Shows safety compliance summary

## 🛠️ Tech Stack
- **Computer Vision:** YOLOv8s (Ultralytics)
- **Backend:** FastAPI, Python
- **Frame Processing:** OpenCV
- **Frontend:** HTML, CSS, JavaScript

## 💡 Use Cases
- PPE compliance monitoring
- Unauthorized access detection
- Worker safety auditing
- Real-time CCTV analysis

## ⚙️ Setup
```bash
pip install ultralytics opencv-python fastapi uvicorn python-multipart
uvicorn main:app --reload
```
Open http://127.0.0.1:8000
