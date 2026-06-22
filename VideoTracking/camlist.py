from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras():
    print(f"Index: {camera_info.index} -> Name: {camera_info.name}")
