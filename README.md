# 🚗 Accident Detection & Alerting System

An end-to-end real-time accident detection system using YOLOv8, DeepSORT, and ConvLSTM. Sends **email and SMS alerts** with video evidence for critical/high-severity incidents detected in surveillance footage.

---

## 📂 Project Structure

```
.
├── main.py                    # Entry point of the system
├── pipeline.py                # Core logic for detection, tracking, classification
├── config.py                 # Configuration: models, classes, paths, constants
├── alerts.py                 # Email and SMS alert functions
├── utils.py                  # Clip generation and frame sequence handling
├── clips/                    # Auto-created to save incident clips
├── models/                   # Folder with YOLO & ConvLSTM models
├── accident_log.csv          # Log file for detected incidents
└── README.md                 
```

---

## 🧠 Key Components

### 1. `config.py`

* Loads all models: YOLOv8 for object + accident detection, ConvLSTM for classification.
* Defines class names, severity levels, alert credentials, log paths, etc.
* Initializes DeepSORT for object tracking.

👉 **Update this file** with:

* Your **Twilio credentials**
* Your **Gmail** and app password
* Model file paths if different

---

### 2. `pipeline.py`

This is the **core engine** that:

* Reads input video/live stream
* Detects objects using YOLOv8 (coco)
* Detects accident zones using YOLOv8 (accident model)
* Classifies incident type with ConvLSTM
* Tracks objects using DeepSORT
* Sends alert via **email** and **SMS** (only once per event)
* Logs every incident to CSV
* Saves pre- and post-incident video clips

---

### 3. `alerts.py`

Handles:

* 📧 Sending email with video attachment
* 📲 Sending SMS with incident summary (via Twilio)

---

### 4. `utils.py`

Utility functions:

* Create frame sequence for ConvLSTM
* Save full-frame clips (3 seconds)
* Save cropped object clips (optional)

---

## 🛠️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/accident-detection-alerts.git
cd accident-detection-alerts
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```txt
opencv-python
numpy
tensorflow
ultralytics
deep_sort_realtime
twilio
```

### 3. Download & Place Models

Put your model files in the `models/` folder:

* `yolov8n.pt` – COCO object detector
* `best.pt` – YOLOv8 accident detector
* `ConvLSTM_best_model.keras` – ConvLSTM classifier

---

## ▶️ How to Run

### Option 1: Process a video file

```bash
python main.py
```

Ensure you update this line in `main.py`:

```python
video_input = "your_video.mp4"  # Path to input video
```

### Option 2: Real-time from webcam

In `main.py`:

```python
video_input = 0  # Use 0 for default webcam
```

---

## 🔄 Flow Summary

1. **YOLOv8 (COCO)** detects objects (car, person, truck, etc.).
2. **DeepSORT** assigns consistent IDs across frames.
3. **YOLOv8 (Accident)** detects accident zones.
4. **ConvLSTM** confirms accident type from past 12 frames.
5. If accident is serious (`High`, `Critical`):

   * Extract 3-second clip (1.5s before & after)
   * Send **email** with video
   * Send **SMS** with details
6. Log all detections to `accident_log.csv`.

---

## 📧 Alert Examples

**Email Subject:**

```
🚨 pedestrian_hit (Critical) Detected
```

**SMS Body:**

```
🚨 pedestrian_hit (Critical)
Object: person (ID 12)
Frame: 275
Clip: ID12_pedestrian_hit.mp4
Time: 2025-05-07 12:35:15
Check your Mail for Video Clip
```

---

## 🔐 Configuration Tips

* **Gmail App Password**: [How to get it](https://support.google.com/accounts/answer/185833?hl=en)
* **Twilio SID/Auth Token**: Set from your Twilio Console.

---

## ✅ Features

* [x] Real-time video processing
* [x] Email + SMS alerts
* [x] Intelligent accident classification
* [x] Per-object tracking
* [x] Logging and video saving
