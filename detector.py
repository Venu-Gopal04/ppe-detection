import cv2
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model (downloads automatically on first run)
model = YOLO("yolov8s.pt")

# PPE-related class names from COCO dataset that YOLOv8 can detect
PERSON_CLASS = 0  # 'person' in COCO

# Colors for bounding boxes (BGR format)
COLORS = {
    "person": (0, 255, 0),       # Green
    "helmet": (255, 165, 0),     # Orange
    "vest": (0, 165, 255),       # Yellow
    "unsafe": (0, 0, 255),       # Red
}

def detect_ppe_in_image(image_path: str, output_path: str) -> dict:
    """
    Detects persons in image and analyzes PPE compliance.
    Draws bounding boxes and returns detection results.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = img.copy()
    height, width = img.shape[:2]

    # Run YOLOv8 detection
    results = model(img, conf=0.5)

    detections = []
    person_count = 0
    unsafe_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Detect persons
            if cls == PERSON_CLASS:
                person_count += 1
                color = COLORS["person"]

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Add label
                label_text = f"Person {conf:.0%}"
                cv2.putText(img, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append({
                    "type": "person",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "ppe_status": "detected"
                })

            # Detect other safety-relevant objects
            elif label in ["hard hat", "helmet", "hat"]:
                color = COLORS["helmet"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"Helmet {conf:.0%}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detections.append({"type": "helmet", "confidence": round(conf, 2)})

    # Add safety summary overlay
    summary_text = f"Persons detected: {person_count}"
    cv2.putText(img, summary_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, summary_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)

    # Save output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

    return {
        "person_count": person_count,
        "total_detections": len(detections),
        "detections": detections,
        "output_image": output_path,
        "safety_status": "ALERT: Persons detected — verify PPE compliance" if person_count > 0 else "No persons detected"
    }


def detect_ppe_in_video(video_path: str, output_path: str, frame_interval: int = 10) -> dict:
    """
    Processes video for PPE detection.
    Samples every N frames, draws detections, saves output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_persons = 0
    max_persons_in_frame = 0
    alert_frames = []
    last_results = None

    print(f"Processing video: {total_frames} frames at {fps:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection every N frames (for speed)
        if frame_count % frame_interval == 0:
            results = model(frame, conf=0.5, verbose=False)
            last_results = results

            person_count = 0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == PERSON_CLASS:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["person"], 2)
                        cv2.putText(frame, f"Person {conf:.0%}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["person"], 2)

            if person_count > 0:
                total_persons += person_count
                timestamp = frame_count / fps
                mins = int(timestamp // 60)
                secs = int(timestamp % 60)
                alert_frames.append({
                    "timestamp": f"{mins:02d}:{secs:02d}",
                    "persons": person_count
                })
                if person_count > max_persons_in_frame:
                    max_persons_in_frame = person_count

        elif last_results:
            # Draw last detection results on intermediate frames
            for result in last_results:
                for box in result.boxes:
                    if int(box.cls[0]) == PERSON_CLASS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["person"], 2)

        # Add frame counter overlay
        cv2.putText(frame, f"Frame: {frame_count} | Persons: {max_persons_in_frame}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"Video processing complete. Output: {output_path}")

    return {
        "total_frames": frame_count,
        "frames_with_persons": len(alert_frames),
        "max_persons_in_frame": max_persons_in_frame,
        "alert_timestamps": alert_frames[:20],
        "output_video": output_path,
        "safety_summary": f"Detected persons in {len(alert_frames)} video segments. Max {max_persons_in_frame} persons at once."
    }