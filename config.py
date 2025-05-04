import os
import cv2
import csv
import datetime
import smtplib
import numpy as np
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client
# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Paths
clip_output_dir = "clips"
log_path = "accident_log.csv"
os.makedirs(clip_output_dir, exist_ok=True)

# Logging CSV header
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'track_id', 'accident_type', 'severity', 'bbox'])


# Class labels and severity
labels = ['collision_with_motorcycle', 'collision_with_stationary_object',
       'drifting_or_skidding', 'fire_or_explosions', 'head_on_collision',
       'negative_samples', 'objects_falling', 'other_crash',
       'pedestrian_hit', 'rear_collision', 'rollover', 'side_collision']

severity_levels = {
    'pedestrian_hit': 'Critical',
    'fire_or_explosions': 'Critical',
    'rollover': 'Critical',
    'head_on_collision': 'Critical',
    'rear_collision': 'High',
    'side_collision': 'High',
    'collision_with_motorcycle': 'High',
    'objects_falling': 'High',
    'drifting_or_skidding': 'Medium',
    'collision_with_stationary_object': 'Medium',
    'other_crash': 'Medium',
    'negative_samples': 'Low'
}


# COCO class names
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Load models
coco_model = YOLO('models/yolov8n.pt')
accident_model = YOLO('models/best.pt')
convlstm_model = load_model('models/ConvLSTM_best_model.keras')
convlstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Twilio Credentials (replace with your actual values or use env vars)
TWILIO_SID = "TWILIO_SID"
TWILIO_AUTH_TOKEN = "TWILIO_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "+TWILIO_PHONE_NUMBER"
TO_PHONE_NUMBER = "TO_PHONE_NUMBER"  # Your verified mobile number

from_email = "your_mail_id@gmail.com"  # replace with your Gmail
app_password = "app_password"  # Gmail App Password