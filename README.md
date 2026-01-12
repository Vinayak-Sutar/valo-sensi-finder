# Valorant Bot Head Detection using YOLOv8

A computer vision project that uses **YOLOv8** to detect and track bot heads in Valorant practice range videos. This can be useful for aim analysis and sensitivity testing.

![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)

## 🎯 Features

- **Train custom YOLO model** on Valorant bot head detection
- **Real-time video inference** with object tracking (ByteTrack)
- **Screen recording** with mouse click timestamps
- **Bullet fire detection** via audio analysis
- **HSV color-based detection** as a fallback method

## 📁 Project Structure

```
sensi_finder/
├── train.py          # Train YOLOv8 model on custom dataset
├── testvid.py        # Run inference on video with tracking
├── test.py           # Run inference on image folder
├── capture.py        # Screen recorder with click timestamps
├── find.py           # HSV color-based target detection
├── bullet.py         # Audio-based bullet fire detection
├── rename.py         # Dataset label preprocessing utility
├── valorant_data.yaml # YOLO dataset configuration
├── dataset/          # Training dataset (images + labels)
└── runs/             # Training outputs and predictions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Vinayak-Sutar/valo-sensi-finder.git
cd valo-sensi-finder

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset as follows:
```
dataset/
├── images/
│   ├── train/    # Training images
│   └── val/      # Validation images
└── labels/
    ├── train/    # YOLO format labels (.txt)
    └── val/      # YOLO format labels (.txt)
```

Each label file should have the format:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all values are normalized (0-1).

### 3. Train the Model

```bash
python train.py
```

Configuration in `train.py`:
- **Epochs**: 50
- **Image size**: 1080
- **Batch size**: 4
- **Model**: YOLOv8n (nano) for speed

### 4. Run Inference on Video

```bash
python testvid.py
```

Edit `testvid.py` to set:
- `model_path`: Path to your trained weights
- `video_path`: Input video file
- `conf_threshold`: Detection confidence (default: 0.1)

## 📊 Dataset Configuration

The `valorant_data.yaml` file defines the dataset:

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 1
names: ['target_bot']
```

## 🛠️ Scripts Overview

| Script | Description |
|--------|-------------|
| `train.py` | Train YOLOv8 on your dataset |
| `testvid.py` | Video inference with ByteTrack tracking |
| `test.py` | Batch inference on image folder |
| `capture.py` | Record screen + log mouse clicks |
| `find.py` | HSV color detection (no ML) |
| `bullet.py` | Detect gunshots from audio |
| `rename.py` | Remap class IDs in label files |

## ⚙️ Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for packages

## 📝 Training Tips

1. **More data = better results** - Aim for 500+ labeled images
2. **Diverse angles** - Include different positions in the practice range
3. **Augmentation** - YOLO applies augmentation automatically
4. **Confidence tuning** - Lower threshold (0.1-0.25) for better recall

## 📄 License

This project is for educational purposes only. Not affiliated with Riot Games.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.
