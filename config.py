import os
import cv2
import csv
import datetime
import smtplib
import numpy as np
import random
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

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
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cattle', 'elephant', 'bear', 'zebra',
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

# Load models with error handling
try:
    coco_model = YOLO('models/yolov8n.pt')
    print("✓ COCO model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load COCO model: {e}")
    raise

try:
    accident_model = YOLO('models/best.pt')
    print("✓ Accident detection model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load accident model: {e}")
    raise

try:
    convlstm_model = load_model('models/ConvLSTM_best_model.keras')
    convlstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✓ ConvLSTM model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load ConvLSTM model: {e}")
    raise

# DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3)

# Twilio Credentials - Load from environment variables
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TO_PHONE_NUMBER = os.getenv("TO_PHONE_NUMBER")

from_email = os.getenv("FROM_EMAIL")
app_password = os.getenv("EMAIL_APP_PASSWORD")

# Validate credentials
def validate_config():
    missing = []
    if not TWILIO_SID:
        missing.append("TWILIO_SID")
    if not TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")
    if not TWILIO_PHONE_NUMBER:
        missing.append("TWILIO_PHONE_NUMBER")
    if not TO_PHONE_NUMBER:
        missing.append("TO_PHONE_NUMBER")
    if not from_email:
        missing.append("FROM_EMAIL")
    if not app_password:
        missing.append("EMAIL_APP_PASSWORD")
    
    if missing:
        print(f"WARNING: Missing environment variables: {', '.join(missing)}")
        print("Alert functionality will be disabled. Set these in .env file or environment.")
        return False
    return True

ALERTS_ENABLED = validate_config()