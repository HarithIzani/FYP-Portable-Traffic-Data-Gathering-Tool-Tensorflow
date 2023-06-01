# || FYP PROOF OF CONCEPT: Pedestrian-Traffic Object Detection for All Environmental Conditions ||
#
# The bounding box technique was inspired by: Evan Juras
# The 2 models were created by: Harith & Hariz (using transfer learning)
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

# Import packages
import os
import numpy as np
import pandas as pd
import cv2
from tflite_runtime.interpreter import Interpreter
from datetime import datetime

# Paths
CWD = os.getcwd() # Current working directory
SOURCE = os.path.join(CWD, 'showcase9min10f.mp4') # Video/cam source
MODEL_DAY = os.path.join(CWD,'TFLite_Model/detect_day.tflite') # Day model
MODEL_NIGHT = os.path.join(CWD,'TFLite_Model/detect_night.tflite') # Night model
LABELS = os.path.join(CWD, 'labels.txt')

# Load Labels
with open(LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Both Models
interpreter_day = Interpreter(MODEL_DAY)
interpreter_day.allocate_tensors()
interpreter_night = Interpreter(MODEL_NIGHT)
interpreter_night.allocate_tensors()

# Model Details
input_details_day = interpreter_day.get_input_details()
output_details_day = interpreter_day.get_output_details()
input_details_night = interpreter_night.get_input_details()
output_details_night = interpreter_night.get_output_details()

floating_model_day = (input_details_day[0]['dtype'] == np.float32)
floating_model_night = (input_details_night[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# We only use day model details day because both models are of same dimensions
height = input_details_day[0]['shape'][1]
width = input_details_day[0]['shape'][2]

# Order for TF2 Model output
boxes_idx, classes_idx, scores_idx = 1, 3, 0

# Loading Source
video = cv2.VideoCapture(SOURCE)
#video = cv2.VideoCapture(0)
#imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Dimension of rpi screen(for resize and coordinate remap)
imW = 800
imH = 470

#Calculating Relative Luminance Function (if fails use img.mean(), we think this function is more accurate)
def luminance(img):
    avg_color_per_row = np.average(img, axis=0)
    BGR = np.average(avg_color_per_row, axis=0)
    return (0.2126 * BGR[2]) + (0.7152 * BGR[1]) + (0.0722 * BGR[0])

# Master Record Dictionary
master_record = {
    'DateTime': [],
    'Lumin': [],
    'Model': [],
    'bicycle': [],
    'bus': [],
    'car': [],
    'motorcycle': [],
    'pedestrian': [],
    'rider': [],
    'train': [],
    'truck': [],
    'other_vehicle': [],
    'other_person': [],
    'trailer': []
}

# Detection/Compilation Loop
while(video.isOpened()):
    # Acquire frame and resize to expected same dimension as model tensors [1xHxWx3]
    ret, frame = video.read()

    # If source cuts out
    if not ret:
      print('Reached the end of the video!')
      break
    
    # Map the opencv BGR to RGB, then resize frame to model input dimensions, will be used to draw bounding boxes
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_frame = np.expand_dims(frame_resized, axis=0) # turning resized frame into numpy arrays

    # Resize frame for openCV drawing
    frame_show = cv2.resize(frame, (imW, imH))

    # If lumen of frame is day criteria (>84)
    lumen = luminance(frame_resized)
    if lumen >= 84:
        # If floating model, normalize pixel value(if model non quantized, this one is)
        input_frame = (np.float32(input_frame) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter_day.set_tensor(input_details_day[0]['index'],input_frame)
        interpreter_day.invoke()

        # Retrieve detection results
        boxes = interpreter_day.get_tensor(output_details_day[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter_day.get_tensor(output_details_day[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter_day.get_tensor(output_details_day[scores_idx]['index'])[0] # Confidence of detected objects

        classes_dictionary = {
            'bicycle':0,
            'bus':0,
            'car':0,
            'motorcycle':0,
            'pedestrian':0,
            'rider':0,
            'train':0,
            'truck':0,
            'other_vehicle':0,
            'other_person':0,
            'trailer':0
        }
        for i,score in enumerate(scores):
            if scores[i] >= 0.5: classes_dictionary[labels[int(classes[i])]] += 1

        #Recording date & time of detection instance
        date_time_instance = datetime.now()
    
        #Updating Master Record
        master_record['DateTime'].append(date_time_instance.strftime("%d/%m/%Y %H:%M:%S"))
        master_record['Lumin'].append(lumen)
        master_record['Model'].append('Day')
        for i,category in enumerate(classes_dictionary.keys()):
            master_record[category].append(classes_dictionary[category])

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] >= float(0.5)) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box, resizing the bouding box to frame show dimensions
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame_show, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame_show, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame_show, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        cv2.putText(frame_show, 'Model: Day ' + 'Lumen: {:.2f}'.format(lumen), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

    if lumen < 84:
        # If floating model, normalize pixel value(if model non quantized, this one is)
        input_frame = (np.float32(input_frame) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter_night.set_tensor(input_details_night[0]['index'],input_frame)
        interpreter_night.invoke()

        # Retrieve detection results
        boxes = interpreter_night.get_tensor(output_details_night[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter_night.get_tensor(output_details_night[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter_night.get_tensor(output_details_night[scores_idx]['index'])[0] # Confidence of detected objects

        classes_dictionary = {
            'bicycle':0,
            'bus':0,
            'car':0,
            'motorcycle':0,
            'pedestrian':0,
            'rider':0,
            'train':0,
            'truck':0,
            'other_vehicle':0,
            'other_person':0,
            'trailer':0
        }
        for i,score in enumerate(scores):
            if scores[i] >= 0.5: classes_dictionary[labels[int(classes[i])]] += 1

        #Recording date & time of detection instance
        date_time_instance = datetime.now()
    
        #Updating Master Record
        master_record['DateTime'].append(date_time_instance.strftime("%d/%m/%Y %H:%M:%S"))
        master_record['Lumin'].append(lumen)
        master_record['Model'].append('Night')
        for i,category in enumerate(classes_dictionary.keys()):
            master_record[category].append(classes_dictionary[category])

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] >= float(0.3)) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box, resizing the bouding box to frame show dimensions
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame_show, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame_show, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame_show, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        cv2.putText(frame_show, 'Model: Night ' + 'Lumen: {:.2f}'.format(lumen), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)
    
    # Display frame with bounding boxes and labels
    cv2.imshow('RaspPi Detector', frame_show)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'): break

# Release memory and cleaning up
video.release()
cv2.destroyAllWindows()

#Converting Master Record to Spreadsheet
master_record_conv = pd.DataFrame.from_dict(master_record)
master_record_conv.to_csv('/home/fyp/Desktop/Master_Record.csv')