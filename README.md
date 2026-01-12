# Valorant Bot Head Detection using YOLOv8

A computer vision project that uses **YOLOv8** to detect and track bot heads in Valorant practice range videos. Useful for aim analysis and sensitivity testing.

![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)

## 🎯 What This Project Does

This project trains a custom YOLOv8 model to detect **bot heads** in Valorant's practice range. The trained model can:
- Detect heads in real-time from video footage
- Track multiple targets using ByteTrack
- Output annotated videos with bounding boxes

## 📁 Project Structure

```
valo-sensi-finder/
├── train.py           # Train YOLOv8 model on custom dataset
├── testvid.py         # Run inference on video with tracking
├── test.py            # Run inference on image folder
├── find.py            # HSV color-based detection (no ML)
├── rename.py          # Dataset label preprocessing utility
├── valorant_data.yaml # YOLO dataset configuration
├── requirements.txt   # Python dependencies
└── runs/
    └── detect/
        ├── train7/    # Training results (curves, metrics)
        └── predict/   # Sample prediction outputs
```

## 📊 Results

Check out the model in action:

https://github.com/user-attachments/assets/YOUR_VIDEO_ID

> Upload your result video to GitHub and replace the link above.

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Vinayak-Sutar/valo-sensi-finder.git
cd valo-sensi-finder
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

```
dataset/
├── images/
│   ├── train/    # Training images
│   └── val/      # Validation images
└── labels/
    ├── train/    # YOLO format labels (.txt)
    └── val/      # YOLO format labels (.txt)
```

Label format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

### 3. Train the Model

```bash
python train.py
```

### 4. Run Inference on Video

Edit `testvid.py` to set your paths, then:
```bash
python testvid.py
```

## ⚙️ Configuration

**train.py settings:**
- Epochs: 50
- Image size: 1080
- Batch size: 4
- Base model: YOLOv8n

**testvid.py settings:**
- `model_path`: Path to trained weights (`runs/detect/train7/weights/best.pt`)
- `video_path`: Your input video
- `conf_threshold`: Detection confidence (default: 0.1)

## 📝 Scripts Overview

| Script | Purpose |
|--------|---------|
| `train.py` | Train YOLOv8 on your dataset |
| `testvid.py` | Video inference with ByteTrack tracking |
| `test.py` | Batch inference on image folder |
| `find.py` | Simple HSV color detection (no ML) |
| `rename.py` | Remap class IDs in label files |

## 🔧 Requirements

- Python 3.8+
- CUDA GPU (recommended)
- See `requirements.txt`

## 📄 License

Educational purposes only. Not affiliated with Riot Games.
