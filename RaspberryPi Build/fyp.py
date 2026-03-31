# || FYP PROOF OF CONCEPT: Pedestrian-Traffic Object Detection for All Environmental Conditions ||
#
# The bounding box technique was inspired by: Evan Juras
# The 2 models were created by: Harith & Hariz (using transfer learning)
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

import os
import sys
import logging
import numpy as np
import pandas as pd
import cv2
from tflite_runtime.interpreter import Interpreter
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
CWD = os.getcwd()
SOURCE      = os.path.join(CWD, 'showcase9min10f.mp4')  # Change to 0 for webcam
MODEL_DAY   = os.path.join(CWD, 'TFLite_Model/detect_day.tflite')
MODEL_NIGHT = os.path.join(CWD, 'TFLite_Model/detect_night.tflite')
LABELS      = os.path.join(CWD, 'labels.txt')
OUTPUT_CSV  = os.path.join(CWD, 'Master_Record.csv')  # Outputs next to script instead of hardcoded RPi path

LUMEN_THRESHOLD = 84    # Luminance value deciding Day vs Night model
CONF_DAY        = 0.5   # Confidence threshold for day model
CONF_NIGHT      = 0.3   # Confidence threshold for night model (more lenient for low-light)
DISPLAY_W       = 800   # RPi screen width
DISPLAY_H       = 470   # RPi screen height
INPUT_MEAN      = 127.5
INPUT_STD       = 127.5
MAX_ERRORS      = 10    # Consecutive frame errors before aborting

# --- Path Validation ---
def validate_paths():
    missing = []
    for name, path in [('Day model', MODEL_DAY), ('Night model', MODEL_NIGHT), ('Labels', LABELS)]:
        if not os.path.isfile(path):
            missing.append(f'  {name}: {path}')
    if missing:
        log.error('Missing required files:\n' + '\n'.join(missing))
        sys.exit(1)

validate_paths()

# --- Load Labels ---
try:
    with open(LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    log.info(f'Loaded {len(labels)} labels')
except Exception as e:
    log.error(f'Failed to load labels: {e}')
    sys.exit(1)

# --- Load Models ---
def load_interpreter(path, name):
    try:
        interp = Interpreter(path)
        interp.allocate_tensors()
        log.info(f'{name} model loaded')
        return interp
    except Exception as e:
        log.error(f'Failed to load {name} model at "{path}": {e}')
        sys.exit(1)

interpreter_day   = load_interpreter(MODEL_DAY,   'Day')
interpreter_night = load_interpreter(MODEL_NIGHT, 'Night')

# --- Model Details ---
input_details_day    = interpreter_day.get_input_details()
output_details_day   = interpreter_day.get_output_details()
input_details_night  = interpreter_night.get_input_details()
output_details_night = interpreter_night.get_output_details()

height = input_details_day[0]['shape'][1]
width  = input_details_day[0]['shape'][2]

# TF2 output tensor order
boxes_idx, classes_idx, scores_idx = 1, 3, 0

# --- Open Video Source ---
def open_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f'Cannot open video source: {source}')
        sys.exit(1)
    log.info(f'Video source opened: {source}')
    return cap

video = open_video(SOURCE)

# --- Relative Luminance (WCAG formula, BGR input) ---
def luminance(img):
    avg_color_per_row = np.average(img, axis=0)
    BGR = np.average(avg_color_per_row, axis=0)
    return (0.2126 * BGR[2]) + (0.7152 * BGR[1]) + (0.0722 * BGR[0])

# --- Master Record ---
CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian',
           'rider', 'train', 'truck', 'other_vehicle', 'other_person', 'trailer']

master_record = {'DateTime': [], 'Lumin': [], 'Model': []}
for cls in CLASSES:
    master_record[cls] = []

# Reusable count dict — reset each frame rather than recreating
classes_dictionary = {cls: 0 for cls in CLASSES}

# Pre-allocated input buffer avoids repeated allocation inside the loop
input_frame_buffer = np.empty((1, height, width, 3), dtype=np.float32)

# --- Unified Detection Function ---
def run_detection(interpreter, input_details, output_details,
                  input_frame, frame_show, conf_threshold, model_name, lumen):
    # Normalize into pre-allocated buffer and run inference
    np.copyto(input_frame_buffer, (np.float32(input_frame) - INPUT_MEAN) / INPUT_STD)
    interpreter.set_tensor(input_details[0]['index'], input_frame_buffer)
    interpreter.invoke()

    boxes   = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores  = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Reset counts for this frame
    for cls in CLASSES:
        classes_dictionary[cls] = 0

    # Count objects above confidence threshold
    for i, score in enumerate(scores):
        if score >= conf_threshold:
            label_name = labels[int(classes[i])]
            if label_name in classes_dictionary:
                classes_dictionary[label_name] += 1

    # Append to master record
    master_record['DateTime'].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    master_record['Lumin'].append(lumen)
    master_record['Model'].append(model_name)
    for cls in CLASSES:
        master_record[cls].append(classes_dictionary[cls])

    # Draw bounding boxes for all detections above threshold
    for i in range(len(scores)):
        if conf_threshold <= scores[i] <= 1.0:
            # np.clip replaces the nested max/min calls
            ymin = int(np.clip(boxes[i][0] * DISPLAY_H, 1, DISPLAY_H))
            xmin = int(np.clip(boxes[i][1] * DISPLAY_W, 1, DISPLAY_W))
            ymax = int(np.clip(boxes[i][2] * DISPLAY_H, 1, DISPLAY_H))
            xmax = int(np.clip(boxes[i][3] * DISPLAY_W, 1, DISPLAY_W))

            cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame_show,
                          (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame_show, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame_show, f'Model: {model_name}  Lumen: {lumen:.2f}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

# --- Detection Loop ---
frame_count = 0
error_count = 0

log.info('Starting detection loop. Press Q to quit.')

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        log.info('Video source ended.')
        break

    try:
        frame_count += 1

        frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_frame   = np.expand_dims(frame_resized, axis=0)
        frame_show    = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        lumen = luminance(frame_resized)

        # BUG FIX: changed second `if` to `elif` so both branches can never both execute.
        # BUG FIX: each branch now passes its own correct confidence threshold (CONF_DAY / CONF_NIGHT)
        #          so night detections are counted at 0.3, not 0.5.
        if lumen >= LUMEN_THRESHOLD:
            run_detection(interpreter_day, input_details_day, output_details_day,
                          input_frame, frame_show, CONF_DAY, 'Day', lumen)
        elif lumen < LUMEN_THRESHOLD:
            run_detection(interpreter_night, input_details_night, output_details_night,
                          input_frame, frame_show, CONF_NIGHT, 'Night', lumen)

        cv2.imshow('RaspPi Detector', frame_show)
        error_count = 0  # reset consecutive error count on successful frame

    except Exception as e:
        error_count += 1
        log.warning(f'Frame {frame_count} error ({error_count}/{MAX_ERRORS}): {e}')
        if error_count >= MAX_ERRORS:
            log.error('Too many consecutive frame errors. Aborting.')
            break

    if cv2.waitKey(1) == ord('q'):
        log.info('User quit.')
        break

# --- Cleanup ---
video.release()
cv2.destroyAllWindows()
log.info(f'Processed {frame_count} frames total.')

# --- Export CSV ---
try:
    master_record_df = pd.DataFrame.from_dict(master_record)
    master_record_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f'Master record saved: {OUTPUT_CSV}')
except Exception as e:
    log.error(f'Failed to save CSV: {e}')
