import cv2
import numpy as np

# Load video
video_path = "vt1.mp4"   # change to your video file
cap = cv2.VideoCapture(video_path)

# HSV color range for target (adjust this after testing)
lower_color = np.array([0, 100, 100])
upper_color = np.array([10, 255, 255])
MIN_AREA = 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours of detected regions
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            target_detected = True

    # Add text
    text = "TARGET DETECTED ✅" if target_detected else "No target"
    color = (0, 255, 0) if target_detected else (0, 0, 255)
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Optional crosshair
    h, w, _ = frame.shape
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)

    # Show both views side by side
    combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Target Detection [Left: Video | Right: Mask]", combined)

    # Press Q to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
