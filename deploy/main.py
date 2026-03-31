"""
Traffic Detection API
Serves both TFLite models (day/night) and runs inference on uploaded image frames.
The portfolio frontend sends frames here and receives bounding boxes + counts as JSON.
"""

import io
import os
import logging
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_DAY   = os.path.join(MODELS_DIR, "detect_day.tflite")
MODEL_NIGHT = os.path.join(MODELS_DIR, "detect_night.tflite")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

LUMEN_THRESHOLD = 84
CONF_DAY        = 0.5
CONF_NIGHT      = 0.3
INPUT_MEAN      = 127.5
INPUT_STD       = 127.5

# TF2 Object Detection API output tensor order
BOXES_IDX, CLASSES_IDX, SCORES_IDX = 1, 3, 0

CLASSES = [
    "bicycle", "bus", "car", "motorcycle", "pedestrian",
    "rider", "train", "truck", "other_vehicle", "other_person", "trailer",
]

# ---------------------------------------------------------------------------
# Globals — populated at startup
# ---------------------------------------------------------------------------
interp_day   = None
interp_night = None
labels       = []
input_h      = 0
input_w      = 0


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global interp_day, interp_night, labels, input_h, input_w

    # Validate files exist
    for label, path in [("Day model", MODEL_DAY), ("Night model", MODEL_NIGHT), ("Labels", LABELS_PATH)]:
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing required file — {label}: {path}")

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    log.info(f"Loaded {len(labels)} labels: {labels}")

    interp_day = Interpreter(MODEL_DAY)
    interp_day.allocate_tensors()

    interp_night = Interpreter(MODEL_NIGHT)
    interp_night.allocate_tensors()

    details = interp_day.get_input_details()
    input_h = details[0]["shape"][1]
    input_w = details[0]["shape"][2]
    log.info(f"Models loaded. Input size: {input_w}x{input_h}")

    yield  # app runs here

    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Traffic Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://harith-izani-portfolio.vercel.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wcag_luminance(rgb_array: np.ndarray) -> float:
    """WCAG relative luminance from an RGB numpy array (H x W x 3)."""
    avg_per_row = np.average(rgb_array, axis=0)   # W x 3
    avg_color   = np.average(avg_per_row, axis=0)  # [R, G, B]
    return float(0.2126 * avg_color[0] + 0.7152 * avg_color[1] + 0.0722 * avg_color[2])


def run_inference(
    interpreter,
    frame_rgb: np.ndarray,
    conf_threshold: float,
) -> tuple[list, dict]:
    """
    Run TFLite inference on a pre-resized RGB frame (H x W x 3).
    Returns (detections list, counts dict).
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data = np.expand_dims(
        (np.float32(frame_rgb) - INPUT_MEAN) / INPUT_STD, axis=0
    )
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()

    boxes   = interpreter.get_tensor(output_details[BOXES_IDX]["index"])[0]
    classes = interpreter.get_tensor(output_details[CLASSES_IDX]["index"])[0]
    scores  = interpreter.get_tensor(output_details[SCORES_IDX]["index"])[0]

    detections = []
    counts     = {cls: 0 for cls in CLASSES}

    for i, score in enumerate(scores):
        if not (conf_threshold <= float(score) <= 1.0):
            continue
        label_idx = int(classes[i])
        if label_idx >= len(labels):
            continue
        label_name = labels[label_idx]
        if label_name in counts:
            counts[label_name] += 1
        detections.append({
            "class": label_name,
            "score": round(float(score), 4),
            "box": {
                "ymin": round(float(boxes[i][0]), 6),
                "xmin": round(float(boxes[i][1]), 6),
                "ymax": round(float(boxes[i][2]), 6),
                "xmax": round(float(boxes[i][3]), 6),
            },
        })

    return detections, counts


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": interp_day is not None,
        "input_size": f"{input_w}x{input_h}",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accept a single image frame (JPEG/PNG), run day or night model
    based on luminance, and return detections + per-class counts.

    Response shape:
    {
      "model_used": "Day" | "Night",
      "luminance": float,
      "detections": [{ "class", "score", "box": {ymin,xmin,ymax,xmax} }],
      "counts": { "car": 3, "pedestrian": 1, ... }
    }
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    # Resize to model input dimensions
    img_resized  = img.resize((input_w, input_h), Image.BILINEAR)
    frame_array  = np.array(img_resized, dtype=np.uint8)  # H x W x 3, RGB

    lumen = wcag_luminance(frame_array)

    if lumen >= LUMEN_THRESHOLD:
        detections, counts = run_inference(interp_day, frame_array, CONF_DAY)
        model_used = "Day"
    else:
        detections, counts = run_inference(interp_night, frame_array, CONF_NIGHT)
        model_used = "Night"

    return {
        "model_used": model_used,
        "luminance":  round(lumen, 2),
        "detections": detections,
        "counts":     counts,
    }
