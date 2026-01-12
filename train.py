from ultralytics import YOLO
import time

def train_yolo():
    # Load pretrained YOLOv8m model
    model = YOLO('yolov8n.pt')

    epochs = 50
    imgsz = 1080
    batch = 4
    device = 0  # GPU id or 'cpu'

    print(f"Starting training: {epochs} epochs, imgsz={imgsz}, batch={batch}, device={device}")
    start_time = time.time()

    # Train the model for all epochs at once
    model.train(
        data='valorant_data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        verbose=True,   # will show batch info, box/loss metrics
        plots=True      # optional: generates loss curves
    )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes.")

if __name__ == '__main__':
    train_yolo()
