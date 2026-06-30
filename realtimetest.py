import cv2
import sys
import numpy as np
from pynput import keyboard
from threading import Thread
import matplotlib.pyplot as plt

sys.path.insert(0, r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Model")
import keypointdetection as kp  # type: ignore

CAM = 1
ENGINE = r"C:\Users\Test\Documents\RTMPose\model.engine"
class Video():
    def __init__(self, cam_index):
        print(f"Initializing stereo camera.")

        self.running = True
        self.cam = cv2.VideoCapture(cam_index)
        self.last_frame = None

        self.prev_ema_score_l = None
        self.prev_ema_score_r = None
        self.prev_ema_coord_l = None
        self.prev_ema_coord_r = None
        self.prev_points3D = None

        self.alpha = 0.675
        self.pose = kp.RTMPose(ENGINE)
    
        self.loadStereoCalib()
        self.setPlotAttr()

    def loadStereoCalib(self):
        # Load calibrated camera features
        fs = cv2.FileStorage(r"C:\Users\Test\Documents\Hand-Tracking\Stereo\stereo.yml", cv2.FILE_STORAGE_READ)
        self.P1 = fs.getNode("P1").mat()
        self.P2 = fs.getNode("P2").mat()
        self.K1 = fs.getNode("K1").mat()
        self.K2 = fs.getNode("K2").mat()
        self.R1 = fs.getNode("R1").mat()
        self.R2 = fs.getNode("R2").mat()
        self.dist1 = fs.getNode("dist1").mat()
        self.dist2 = fs.getNode("dist2").mat()
        fs.release()

    def setPlotAttr(self):
        plt.ion()

        self.fig = plt.figure()

        frame_size = 75
        ax = self.fig.add_subplot(111, projection = '3d')
        
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 30)  

        ax.zaxis.set_inverted(True)
        ax.view_init(elev = 90, azim = 90, roll = 0)   
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 

        self.scatter = ax.scatter([], [], [], color = (196 / 255, 12 / 255, 27 / 255), s = 15, clip_on = True)
        self.lines = []
        for _ in kp.HAND_SKELETON:
            line, = ax.plot([], [], [], 'b-', clip_on = True)
            self.lines.append(line)

        plt.title("3D Mapped Hand Skeleton Keypoints (In Centimeters)")

    def ema(self, arr, alpha,  prev):
        smoothed = arr * alpha + (1 - alpha) * prev
        return smoothed
            
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

                self.left_coords = kp_frame[0][0]
                self.right_coords = kp_frame[0][1]
                left_score = kp_frame[1][0]
                right_score = kp_frame[1][1]
          
                if self.prev_ema_coord_l is None:
                    self.prev_ema_coord_l = self.left_coords.copy()
                    self.prev_ema_coord_r = self.right_coords.copy()
                else:
                    for i in range(21):
                        # Convert keypoint 2D coordinate list to numpy array then back to list to do ema computation
                        left_coord_arr = np.array(self.left_coords[i])
                        right_coord_arr = np.array(self.right_coords[i])
                        prev_ema_coord_l_arr = np.array(self.prev_ema_coord_l[i])
                        prev_ema_coord_r_arr = np.array(self.prev_ema_coord_r[i])

                        self.prev_ema_coord_l[i] = self.ema(left_coord_arr, self.alpha, prev_ema_coord_l_arr).tolist()
                        self.prev_ema_coord_r[i] = self.ema(right_coord_arr, self.alpha, prev_ema_coord_r_arr).tolist()

                if self.prev_ema_score_l is None:
                    self.prev_ema_score_l = left_score.copy()
                    self.prev_ema_score_r = right_score.copy()
                else:
                    self.prev_ema_score_l = self.ema(left_score, self.alpha, self.prev_ema_score_l)
                    self.prev_ema_score_r = self.ema(right_score, self.alpha, self.prev_ema_score_r)

                left_frame = self.pose.draw_hand(left, self.prev_ema_coord_l, self.prev_ema_score_l)
                right_frame = self.pose.draw_hand(right, self.prev_ema_coord_r, self.prev_ema_score_r)

                cv2.imshow("Left Camera", left_frame)
                cv2.imshow("Right Camera", right_frame)
                #self.plot3D()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit()
                    break
    
    def plot3D(self):
        count = 0
        for i in range(21):
            if (self.prev_ema_score_l[i] < self.pose.conf or self.prev_ema_score_r[i] < self.pose.conf):
                count += 1
        if count >= 5:
            return
            
        pts_left = []
        pts_right = []

        for i in range(21):
            pts_left.append(self.left_coords[i])
            pts_right.append(self.right_coords[i])

        pts_left = np.asarray(pts_left, dtype = np.float32)
        pts_right = np.asarray(pts_right, dtype = np.float32)

        pts_left_cv = pts_left[:, np.newaxis, :]
        pts_right_cv = pts_right[:, np.newaxis, :]

        # Undistort and rectify points
        pts_left_rect = cv2.undistortPoints(pts_left_cv, self.K1, self.dist1, R=self.R1, P=self.P1)
        pts_right_rect = cv2.undistortPoints(pts_right_cv, self.K2, self.dist2, R=self.R2, P=self.P2)

        # Flatten back to (N, 2)
        pts_left_rect = pts_left_rect.squeeze(1)
        pts_right_rect = pts_right_rect.squeeze(1)

        # Triangulate using your corrected coordinates
        points4D = cv2.triangulatePoints(self.P1, self.P2, pts_left_rect.T, pts_right_rect.T)      # Produces output of size (X, Y, Z, W)                                      
        points3D = (points4D[:3] / points4D[3]) * 100                                                              # Transpose to get shape (N, 3) instead of (3, N) and multiply by 100 for cm
        points3D = np.squeeze(points3D)

        if self.prev_points3D is None:
            self.prev_points3D = points3D.copy()
        else:
            self.prev_points3D = self.ema(points3D, self.alpha - 0.2, self.prev_points3D)

        self.scatter._offsets3d = (self.prev_points3D[:, 0], self.prev_points3D[:, 1], self.prev_points3D[:, 2])
        
        for line, (start, end) in zip(self.lines, kp.HAND_SKELETON):
            line.set_data([self.prev_points3D[start, 0], self.prev_points3D[end, 0]], [self.prev_points3D[start, 1], self.prev_points3D[end, 1]])
            line.set_3d_properties([self.prev_points3D[start, 2], self.prev_points3D[end, 2]])

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