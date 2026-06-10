import cv2
from pynput import keyboard

class Vid2Frame():
    def __init__(self, cam_index):
        print("Initializing Camera.")
    
        self.cam = cv2.VideoCapture(cam_index)
        self.fps = 60
        self.frame_rate_dur = int(1000 / self.fps)

        self.running = True
        self.listener = keyboard.Listener(on_press = self.quit)
        self.listener.start()

    def quit(self): self.running = False
    
    def take_frame(self):
        ret, frame = self.cam.read()
        if ret:
            frame_dir = "VideoTracking/frame.png"
            cv2.imwrite(frame_dir, frame)
    
    def capture(self):
        while self.running:
            self.take_frame()
            cv2.waitKey(self.frame_rate_dur)
        
        self.listener.stop()
        self.cam.release()

video = Vid2Frame(0)
video.capture()