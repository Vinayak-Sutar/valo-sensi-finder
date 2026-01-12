from ultralytics import YOLO
import cv2
import time
import os

def test_on_video():
    # ====== CONFIGURATION ======
    model_path = "runs/detect/train7/weights/best.pt"  # Trained model
    video_path = "video_samples/vv.mp4"            # Input video path
    output_dir = "runs/detect/video_results"          # Folder for output
    conf_threshold = 0.1                             # Confidence threshold
    imgsz = 1080                                      # Resize for inference
    device = 0                                        # GPU: 0 | CPU: 'cpu'

    # ====== LOAD MODEL ======
    print("\n🔹 Loading model...")
    start_time = time.time()
    model = YOLO(model_path)
    print(f"✅ Model loaded in {time.time() - start_time:.2f}s")

    # ====== TEST ON VIDEO ======
    print(f"\n🎥 Starting inference on: {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=video_path,
        tracker="bytetrack.yaml",
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        save=True,
        save_txt=False,
        project=output_dir,
        name="run1",
        show=True  # set False if you don’t want preview window
    )

    print("\n✅ Video inference complete!")
    print(f"📁 Output saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    test_on_video()
