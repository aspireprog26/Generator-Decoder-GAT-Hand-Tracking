import cv2
import sys
from pynput import keyboard
from threading import Thread

sys.path.insert(1, r"C:\Users\Test\Documents\Hand-Tracking\Model")
import keypointdetection as kp  # type: ignore

CAM = 0
ENGINE = r"C:\Users\Test\Documents\RTMPose\model.engine"
class Video():
    def __init__(self, cam_index, camera):
        print(f"Initializing {camera} camera.")

        self.running = True
        self.cam = cv2.VideoCapture(cam_index)
        self.last_frame = None

        self.pose = kp.RTMPose(ENGINE)

    def take_frame(self):
        while self.running:
            ret, frame = self.cam.read()
            if ret:
                self.last_frame = frame

    def get_frame(self):
        while self.running:
            if self.last_frame is not None:
                kp_frame = self.pose.get_keypoints(self.last_frame, self.last_frame)[1][0]
                cv2.imshow("Hand Tracking 2D", kp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit()
                    break

    def start(self):
        take_frame_thread = Thread(target = self.take_frame, daemon = True)
        take_frame_thread.start()
        self.get_frame()

    def quit(self):
        self.running = False
        self.cam.release()
        cv2.destroyAllWindows()

left = Video(CAM, "left")

def quit_streams():
    left.quit()
    listener.stop()
    
listener = keyboard.Listener(on_press = quit_streams)
listener.start()
left.start()
