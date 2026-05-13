# 🚦 AI Traffic Monitoring System (YOLOv11 + ByteTrack + CI/CD)

##  Overview

This project is an end-to-end **AI-powered traffic monitoring system** that analyzes CCTV video streams in real-time using Computer Vision and Deep Learning.

It detects vehicles, tracks them across frames, analyzes road conditions, and classifies traffic congestion levels. The system is deployed as a Flask web application with CI/CD automation.

---

##  Key Features

- 🚗 Vehicle detection using YOLOv11 (car, bus, truck, motorcycle)
- 🎯 Multi-object tracking using ByteTrack
- 🛣️ Road segmentation (road, lanes, pedestrian crossing)
- 📊 Real-time traffic analytics (flow, occupancy, congestion level)
- 🌐 Web dashboard using Flask
- ☁️ Cloud deployment using Google Colab + Cloudflare tunnel
- 🔄 CI/CD pipeline using GitHub Actions
- 💾 Data storage using Supabase

---

##  System Architecture

The system follows this pipeline:

Input CCTV Video  
→ YOLOv11 Vehicle Detection  
→ Bounding Boxes + Confidence Scores  
→ ByteTrack Tracking  
→ Traffic Feature Extraction  
→ Analytics Engine  
→ Flask Dashboard Output  

---

##  Modules

### 1. Vehicle Detection (YOLOv11)
- Detects vehicles per frame
- Outputs:
  - Bounding boxes
  - Class labels
  - Confidence scores

---

### 2. Object Tracking (ByteTrack)

- Multi-object tracking using ByteTrack (Ultralytics implementation)
- Maintains consistent object IDs across frames
- Handles occlusion and re-identification

---

### 3. Road Segmentation (YOLOv11-seg)
- Detects:
  - Road surface
  - Lane markings
  - Pedestrian crossings
- Produces pixel-level segmentation masks

---

### 4. Traffic Analytics Engine
- Vehicle counting per frame
- Flow estimation (vehicles/hour)
- Road occupancy percentage
- Congestion classification:
  - Free Flow
  - Moderate Traffic
  - Heavy Congestion

---

## Tech Stack

- Python
- YOLOv11 / YOLOv11-seg (Ultralytics)
- ByteTrack (Ultralytics implementation)
- OpenCV
- Flask (Web Application)
- HTML / CSS / JavaScript
- Supabase (Database)
- GitHub Actions (CI pipeline)
- Google Colab (GPU training environment)
- Cloudflare Tunnel (temporary demo for local Flask exposure)

---


## CI/CD Pipeline

This project implements Continuous Integration (CI) using GitHub Actions.

On every push:
- Code checkout
- Python environment setup
- Dependency installation
- Syntax validation
- Unit tests execution

Continuous Deployment (CD) is planned for future enhancement (e.g., AWS / Render deployment).

## Deployment

The application is run locally using Flask and exposed temporarily using Cloudflare Tunnel for demonstration purposes.

Note: This is not production deployment.

---

## 📊 Results

- High accuracy vehicle detection using YOLOv11
- Robust tracking under occlusion
- Real-time traffic classification
- Stable performance on CCTV data

---

## 📁 Project Structure

Vehicles_models/
│
├── app.py
├── config.py
├── pipeline.py
├── database.py
├── utils.py
├── frontend/
├── tests/
├── .github/workflows/
└── requirements.txt

- app.py → Flask backend API
- pipeline.py → AI inference pipeline (YOLO + tracking + analytics)
- config.py → system configuration
- database.py → Supabase integration
- utils.py → helper functions
- frontend/ → web interface
- tests/ → CI test suite

-----
## Key Achievements

- Built end-to-end AI system for real-time traffic analysis
- Trained YOLOv11 detection model on CCTV vehicle dataset
- Implemented road scene segmentation using YOLOv11-seg
- Integrated multi-object tracking for vehicle trajectory analysis
- Designed traffic congestion classification system
- Automated CI pipeline using GitHub Actions

## Note: 
This project was built as a final Master’s thesis project and enhanced with CI/CD practices for production-level deployment.


