"""
YOLO Object Detection Engine for Autonomous Driving.
Wraps YOLOv8 / YOLOv9 / custom fine-tuned weights for real-time perception.
Detects cars, pedestrians, cyclists, trucks, and traffic signs with bounding box coordinates and confidences.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Try importing ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class YOLODetector:
    TARGET_CLASSES = {
        0: "pedestrian",
        1: "cyclist",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic_light",
        11: "traffic_sign"
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.40, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None

        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"[YOLO] Loaded model weights from {model_path}")
            except Exception as e:
                print(f"[YOLO Warning] Could not load {model_path} ({e}). Using synthetic road detector fallback.")
                self.model = None
        else:
            print("[YOLO Notice] ultralytics package not detected. Operating in simulated inference mode.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a BGR image frame.
        Returns list of detections: [x1, y1, x2, y2, conf, class_name]
        """
        h, w = frame.shape[:2]
        detections = []

        if self.model is not None:
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    name = self.TARGET_CLASSES.get(cls_id, self.model.names.get(cls_id, "object"))
                    detections.append({
                        "bbox": [round(c, 1) for c in xyxy],
                        "confidence": round(conf, 3),
                        "class_id": cls_id,
                        "class_name": name
                    })
        else:
            # Fallback perception using color segmentation and contour heuristic
            # Detect red/dark vehicles and bright pedestrians in lower half
            road_roi = frame[int(h*0.45):, :]
            gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 75, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 800 < area < 40000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bh / float(bw)
                    real_y = y + int(h * 0.45)
                    cls_name = "pedestrian" if aspect > 1.8 else "car"
                    detections.append({
                        "bbox": [x, real_y, x + bw, real_y + bh],
                        "confidence": 0.89,
                        "class_id": 0 if cls_name == "pedestrian" else 2,
                        "class_name": cls_name
                    })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = (0, 0, 255) if cls_name == "pedestrian" else (0, 255, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out

if __name__ == "__main__":
    from data.dataset_loader import PerceptionDatasetLoader
    frame, _ = PerceptionDatasetLoader.generate_synthetic_frame(0)
    detector = YOLODetector()
    dets = detector.detect(frame)
    print(f"Detected {len(dets)} objects:", dets)
