# 📊 AI-Powered Real-Time Classroom Monitoring System

An automated, high-precision attendance tracking system that leverages Edge AI to monitor student presence, detect unauthorized exits (“bunking”), and ensure complete data privacy.

---

## 🚀 Overview

Traditional attendance systems only capture presence at a single point in time, making them ineffective against proxy attendance and mid-class exits.

This project introduces a real-time monitoring solution that continuously tracks student presence throughout the lecture using computer vision and intelligent temporal logic.

---

## ✨ Key Features

### Dual-Model Pipeline

Combines YOLOv8 (Nano variant) for real-time person detection and InsightFace (ArcFace) for accurate student identification.

### Temporal Logic Engine

Implements a smart tracking mechanism with a **15-minute grace period** to differentiate between temporary exits and unauthorized absences.

### Edge-First Architecture

All processing is performed locally, ensuring:

* 🔒 100% data privacy
* ⚡ Low latency (~22 FPS)
* 🚫 No cloud dependency

### Faculty Dashboard

Built with Streamlit for:

* Live classroom monitoring
* Instant attendance reports
* Database management

---

## 🛠️ Tech Stack

| Component   | Technology            |
| ----------- | --------------------- |
| Detection   | YOLOv8 (Nano)         |
| Recognition | InsightFace (ArcFace) |
| Database    | SQLite (Local)        |
| Frontend    | Streamlit             |
| Programming | Python 3.x            |

---

## 🧠 System Architecture

```
Video Input → Live classroom feed  
     ↓
Detection Layer → YOLOv8 detects all individuals  
     ↓
Recognition Layer → InsightFace identifies students  
     ↓
Tracking Engine → Maintains presence state over time  
     ↓
Temporal Logic → Flags bunking based on absence duration  
     ↓
Dashboard → Displays real-time insights  
```

## 📂 Project Structure

```
├── data/database/        # SQLite database files  
├── Yolov8n.pt            # Pre-trained model  
├── src/dashboard/        # Streamlit app  
├── src/main.py           # Entry point  
├── requirements.txt      # Dependencies  
└── README.md             # Project documentation  
```
## 🔐 Privacy & Security

* All facial data is processed locally (Edge AI)
* No cloud storage or external API calls
* Provides full control over sensitive biometric data.


