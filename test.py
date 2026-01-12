from ultralytics import YOLO
import time
import os

def test_yolo():
    # ====== CONFIGURATION ======
    model_path = "runs/detect/train4/weights/best.pt"  # Path to trained weights
    test_folder = "dataset/images/val"              # Path to your test images folder
    conf_threshold = 0.25                             # Detection confidence
    imgsz = 1280                                      # Resize for inference
    device = 0                                        # GPU (0) or CPU ('cpu')

    # ====== LOAD MODEL ======
    print("\n🔹 Loading model...")
    start_time = time.time()
    model = YOLO(model_path)
    print(f"✅ Model loaded in {time.time() - start_time:.2f}s")

    # ====== TEST ON FOLDER ======
    print(f"\n🔹 Starting inference on folder: {test_folder}")
    print(f"🔸 Confidence threshold: {conf_threshold}")
    print(f"🔸 Image size: {imgsz}")
    print(f"🔸 Device: {'GPU' if device == 0 else 'CPU'}")

    start_infer_time = time.time()
    results = model.predict(
        source=test_folder,
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        show=False,
        save=True,
        stream=False,
        verbose=True
    )

    # ====== SUMMARY ======
    total_time = time.time() - start_infer_time
    save_dir = model.predictor.save_dir if hasattr(model, "predictor") else "runs/detect/predict"

    print(f"\n✅ Inference complete!")
    print(f"📁 Results saved in: {os.path.abspath(save_dir)}")
    print(f"⏱ Total inference time: {total_time:.2f} seconds")

    # If you want to view results immediately
    print("\n🔍 Sample prediction files:")
    for i, f in enumerate(os.listdir(save_dir)):
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.mp4')):
            print(f"  {i+1}. {f}")
            if i >= 4:  # show only a few
                break

if __name__ == "__main__":
    test_yolo()
