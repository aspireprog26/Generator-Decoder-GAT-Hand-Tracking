import cv2
import sys
import numpy as np
from pynput import keyboard
from threading import Thread
import matplotlib.pyplot as plt

sys.path.insert(1, r"C:\Users\Test\Documents\Hand-Tracking\Model")
import keypointdetection as kp  # type: ignore

CAM = 1
ENGINE = r"C:\Users\Test\Documents\RTMPose\model.engine"
class Video():
    def __init__(self, cam_index):
        print(f"Initializing stereo camera.")

        self.running = True
        self.cam = cv2.VideoCapture(cam_index)

        self.last_frame = None
        self.avg_left_coord = None
        self.avg_right_coord = None

        self.alpha = 0.6
        self.pose = kp.RTMPose(ENGINE)
    
        self.loadStereoCalib()
        self.setPlotAttr()

    def loadStereoCalib(self):
        # Load calibrated camera features
        fs = cv2.FileStorage(r"C:\Users\Test\Documents\Hand-Tracking\Stereo\stereo.yml", cv2.FILE_STORAGE_READ)
        self.P1 = fs.getNode("P1").mat()
        self.P2 = fs.getNode("P2").mat()
        fs.release()

    def setPlotAttr(self):
        plt.ion()

        self.fig = plt.figure()
        frame_size = 30
        ax = self.fig.add_subplot(111, projection = '3d')
        
        ax.set_xlim(-frame_size, frame_size)
        ax.set_ylim(-frame_size, frame_size)
        ax.set_zlim(0, frame_size)  

        ax.zaxis.set_inverted(True)
        ax.view_init(elev = 20, azim = 50, roll = 0)   
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 

        self.scatter = ax.scatter([], [], [], color = (196 / 255, 12 / 255, 27 / 255), s = 15)
        self.lines = []
        for _ in kp.HAND_SKELETON:
            line, = ax.plot([], [], [], 'b-')
            self.lines.append(line)

        plt.title("3D Mapped Hand Skeleton Keypoints (In Centimeters)")

    def ema(self, arr, alpha, axis):
        arr = np.asarray(arr)
        x = np.moveaxis(arr, axis, 0)

        y = np.empty_like(x, dtype = float)
        y[0] = x[0]

        for i in range(1, x.shape[0]):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]

        return np.moveaxis(y, 0, axis)
            
    def take_frame(self):
        while self.running:
            ret, frame = self.cam.read()
            if ret:
                self.last_frame = frame

    def get_frame(self):
        while self.running:
            if self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                half = w // 2

                left  = self.last_frame[:, :half]
                right = self.last_frame[:, half:]
                kp_frame = self.pose.get_keypoints(left, right)

                avg_score_right = self.ema(kp_frame[1][1], self.alpha, 0)                
                avg_score_left = self.ema(kp_frame[1][0], self.alpha, 0)

                self.avg_left_coord = kp_frame[0][0]
                self.avg_right_coord = kp_frame[0][1]
 
                for i in range(len(self.avg_left_coord)):
                    left_kps = self.avg_left_coord[i]
                    right_kps = self.avg_right_coord[i]
                    for kp_l, kp_r in zip(left_kps, right_kps):
                        kp_l = self.alpha * kp_l + (1 - self.alpha) * kp_l
                        kp_r = self.alpha * kp_r + (1 - self.alpha) * kp_r

                left_frame = self.pose.draw_hand(left, self.avg_left_coord, avg_score_left)
                right_frame = self.pose.draw_hand(right, self.avg_right_coord, avg_score_right)

                cv2.imshow("Left Camera", left_frame)
                cv2.imshow("Right Camera", right_frame)
                self.plot3D()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit()
                    break
    
    def plot3D(self):
        pts_left = []
        pts_right = []

        for i in range(len(self.avg_left_coord)):
            pts_left.append(self.avg_left_coord[i])
            pts_right.append(self.avg_right_coord[i])
                
        pts_left = np.array(pts_left)
        pts_right = np.array(pts_right)

        # Obtain 4D points and scale to 3D
        points4D = cv2.triangulatePoints(self.P1, self.P2, pts_left.T, pts_right.T)      # Produces output of size (X, Y, Z, W)
        points3D = points4D[:3] / points4D[3]                                      
        points3D = (points3D.T) * 100                                                    # Transpose to get shape (N, 3) instead of (3, N) and multiply by 100 for cm

        self.scatter._offsets3d = (points3D[:, 0], points3D[:, 1], points3D[:, 2])
        for line, (start, end) in zip(self.lines, kp.HAND_SKELETON):
            line.set_data([points3D[start, 0], points3D[end, 0]], [points3D[start, 1], points3D[end, 1]])
            line.set_3d_properties([points3D[start, 2], points3D[end, 2]])
            
        self.fig.canvas.draw_idle()         # Redraw when ready
        self.fig.canvas.flush_events()      # Process pending GUI events

    def start(self):
        take_frame_thread = Thread(target = self.take_frame, daemon = True)
        take_frame_thread.start()
        self.get_frame()
        plt.show()

    def quit(self):
        self.running = False
        self.cam.release()
        cv2.destroyAllWindows()

left = Video(CAM)

def quit_streams():
    left.quit()
    listener.stop()
    
listener = keyboard.Listener(on_press = quit_streams)
listener.start()
left.start()