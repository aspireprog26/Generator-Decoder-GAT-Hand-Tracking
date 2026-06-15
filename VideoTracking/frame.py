import cv2
from queue import Queue
from pynput import keyboard
from threading import Thread

CAM = 0

class Video:
    def __init__(self, cam_index, camera):
        print(f"Initializing {camera} camera.")

        self.running = True
        self.cam = cv2.VideoCapture(cam_index)
        self.frame_queue = Queue(maxsize = 1)       # Only hold one frame at a time

    def take_frame(self):
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put(frame)

    def get_frame(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow("Hand Tracking 2D", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    self.cam.release()
                    cv2.destroyAllWindows()
                    break

    def start(self):
        take_frame_thread = Thread(target = self.take_frame, daemon = True)
        self.get_frame()

left = Video(CAM, "left")
left.start()
