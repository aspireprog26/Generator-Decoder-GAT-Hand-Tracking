import os  # noqa: I001
import cv2

video_path = "/home/miket/Documents/StereoDataset/Noisy/3.mp4"
output_dir = "/home/miket/Documents/StereoDataset/Noisy"

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_dir, f"{frame_count:04d}.jpg")
    cv2.imwrite(filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames.")
