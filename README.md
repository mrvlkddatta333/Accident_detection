# 🚨 Accident Detection System (YOLOv8 + DeepSORT + ConvLSTM)

This project is a hybrid accident detection system combining real-time object detection, tracking, and video-based classification. It identifies accidents in videos and classifies their type and severity using deep learning.

## 🔧 Features

* **YOLOv8** for detecting road objects (vehicles, people, etc.)
* **Custom YOLOv8 model** for detecting accidents/non-accidents
* **DeepSORT** for multi-object tracking with unique IDs
* **ConvLSTM** for accident type classification using video sequences
* **Logging** (CSV format) of accident events with details
* **Video clip saving** for detected accident intervals
* **email/SMS alerts** (can be configured in `alerts.py`)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/accident-detection-system.git
cd accident-detection-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Add Model Weights

* Place your trained **YOLOv8 weights** (e.g., `best.pt`) in the appropriate path.
* Include your trained **ConvLSTM model** (e.g., `convlstm_model.keras`) as defined in `config.py`.

---

### 4. Run the Full Pipeline

```bash
python main.py
```

By default, this runs on the input video path configured in `main.py`.

If using a webcam, update `video_input = 0` in `main.py`.

---

## 📁 Output

* Annotated output video (`output_YYYYMMDD_HHMMSS.mp4`)
* Logged events in CSV (`accident_log.csv`)
* Saved accident clips in `/clips/` folder

---

## 🔧 Configuration

All paths, thresholds, model configs can be adjusted in:

```python
config.py
```

For alerts, update the placeholders in:

```python
alerts.py
```

---

## 🧠 Model Architecture

* **YOLOv8**: Detect objects and accident regions in each frame.
* **DeepSORT**: Track objects across frames and assign unique IDs.
* **ConvLSTM**: Predict accident type based on last `N` frames around detection.

---
