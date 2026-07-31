import json
import sys
from pathlib import Path
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pynput import keyboard

sys.path.insert(0, r"C:\Users\Test\Documents\Hand-Tracking-2\Keypoints")
sys.path.insert(0, r"C:\Users\Test\Documents\Hand-Tracking-2\Model")
sys.path.insert(0, r"C:\Users\Test\Documents\Hand-Tracking-2\Dataset")


import keypointdetection as kp
from handedgeindex import hand_edge_index
from model import AnatomyModel
from posttrainerreg import Trainer
from pretrainer import Trainer as PreTrainer

CAM = 1
ENGINE = r"C:\Users\Test\Documents\RTMPose\model.engine"


class Video:
    def __init__(self, cam_index):
        print("Initializing stereo camera.")

        self.running = True
        self.cam = cv2.VideoCapture(cam_index)
        self.last_frame = None

        self.prev_ema_score_l = None
        self.prev_ema_score_r = None
        self.prev_ema_coord_l = None
        self.prev_ema_coord_r = None
        self.left_coords = None
        self.right_coords = None
        self.prev_points3D = None

        self.alpha = 0.7
        self.PALM = [0, 1, 5, 9, 13, 17]
        self.pose_rtm = kp.RTMPose(ENGINE)
        self.pose_mp = kp.MediaPipe()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.standardize = Trainer().standardize
        self.feats = PreTrainer().features
        self.edge_index = hand_edge_index.to(device)
        self.batch = torch.zeros(21, dtype=torch.long, device=device)

        with open(
            "/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r"
        ) as f:
            configs = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnatomyModel(
            configs["input_size"],
            configs["decoder_hidden_size"],
            configs["decoder_output_size"],
            configs["decoder_dropout"],
            generator=False,
        ).to(device)

        weights = torch.load(
            (Path(configs["model_dir"]) / f"{configs['decoder_model_name']}reg"),
            weights_only=True,
            map_location=device,
        )

        self.model.load_state_dict(weights)
        self.model.eval()

        self.loadStereoCalib()
        self.setPlotAttr()

    def loadStereoCalib(self):
        # Load calibrated camera features
        fs = cv2.FileStorage(
            r"C:\Users\Test\Documents\Hand-Tracking-2\Stereo\stereo.yml",
            cv2.FILE_STORAGE_READ,
        )

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
        frame_size = 65

        self.fig1 = plt.figure()
        ax1 = self.fig1.add_subplot(111, projection="3d")
        ax1.set_xlim(-frame_size, frame_size)
        ax1.set_ylim(-frame_size, frame_size)
        ax1.set_zlim(0, frame_size)
        ax1.zaxis.set_inverted(True)
        ax1.view_init(elev=220, azim=130, roll=0)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        self.scatter1 = ax1.scatter(
            [], [], [], color=(196 / 255, 12 / 255, 27 / 255), s=15, clip_on=True
        )
        self.lines1 = []
        for _ in kp.HAND_SKELETON:
            (line,) = ax1.plot([], [], [], "b-", clip_on=True)
            self.lines1.append(line)
        plt.figure(self.fig1.number)
        plt.title("Raw Projected Hand Keypoints")

        self.fig2 = plt.figure()
        ax2 = self.fig2.add_subplot(111, projection="3d")
        ax2.set_xlim(-frame_size, frame_size)
        ax2.set_ylim(-frame_size, frame_size)
        ax2.set_zlim(0, frame_size)
        ax2.zaxis.set_inverted(True)
        ax2.view_init(elev=220, azim=130, roll=0)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        self.scatter2 = ax2.scatter(
            [], [], [], color=(12 / 255, 196 / 255, 27 / 255), s=15, clip_on=True
        )
        self.lines2 = []
        for _ in kp.HAND_SKELETON:
            (line,) = ax2.plot([], [], [], "g-", clip_on=True)
            self.lines2.append(line)
        plt.figure(self.fig2.number)
        plt.title("Corrected 3D Projected Hand Keypoints")

    def ema(self, arr, alpha, prev):
        smoothed = arr * alpha + (1 - alpha) * prev
        return smoothed

    def take_frame(self):
        while self.running:
            ret, frame = self.cam.read()
            if ret:
                self.last_frame = frame

    def get_frame_rtm(self):
        while self.running:
            if self.last_frame is not None:
                _, w = self.last_frame.shape[:2]
                half = w // 2

                left = self.last_frame[:, :half]
                right = self.last_frame[:, half:]
                kp_frame = self.pose_rtm.get_keypoints(left, right)

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

                        self.prev_ema_coord_l[i] = self.ema(
                            left_coord_arr, self.alpha, prev_ema_coord_l_arr
                        ).tolist()
                        self.prev_ema_coord_r[i] = self.ema(
                            right_coord_arr, self.alpha, prev_ema_coord_r_arr
                        ).tolist()

                if self.prev_ema_score_l is None:
                    self.prev_ema_score_l = left_score.copy()
                    self.prev_ema_score_r = right_score.copy()
                else:
                    self.prev_ema_score_l = self.ema(
                        left_score, self.alpha, self.prev_ema_score_l
                    )
                    self.prev_ema_score_r = self.ema(
                        right_score, self.alpha, self.prev_ema_score_r
                    )

                left_frame = self.pose_rtm.draw_hand(
                    left, self.prev_ema_coord_l, self.prev_ema_score_l
                )
                right_frame = self.pose_rtm.draw_hand(
                    right, self.prev_ema_coord_r, self.prev_ema_score_r
                )

                cv2.imshow("Left Camera", left_frame)
                cv2.imshow("Right Camera", right_frame)

                self.plot3D("rtm")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.quit()
                    break

    def get_frame_mp(self):
        while self.running:
            if self.last_frame is not None:
                _, w = self.last_frame.shape[:2]
                half = w // 2

                left = self.last_frame[:, :half]
                right = self.last_frame[:, half:]

                self.left_coords = self.pose_mp.get_keypoints(left)
                self.right_coords = self.pose_mp.get_keypoints(right)

                if self.prev_ema_coord_l is None:
                    self.prev_ema_coord_l = self.left_coords.copy()
                    self.prev_ema_coord_r = self.right_coords.copy()
                else:
                    self.prev_ema_coord_l = self.ema(
                        self.left_coords, self.alpha, self.prev_ema_coord_l
                    )
                    self.prev_ema_coord_r = self.ema(
                        self.right_coords, self.alpha, self.prev_ema_coord_r
                    )

                self.left_frame = self.pose_mp.draw_hand(self.prev_ema_coord_l, left)
                self.right_frame = self.pose_mp.draw_hand(self.prev_ema_coord_r, right)

                cv2.imshow("Left Camera", self.left_frame)
                cv2.imshow("Right Camera", self.right_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.quit()
                    break

    def computeEma(self, points3D):
        if self.prev_points3D is None:
            prev_points3D = points3D.copy()
        else:
            prev_points3D = self.ema(points3D, self.alpha, self.prev_points3D)
        return prev_points3D

    def plot3D(self, mode):
        while self.running:
            pts_left = None
            pts_right = None
            if mode == "rtm":
                pts_left = []
                pts_right = []

                for i in range(21):
                    pts_left.append(self.left_coords[i])
                    pts_right.append(self.right_coords[i])
            else:
                if self.left_coords is not None and self.right_coords is not None:
                    pts_left = self.left_coords
                    pts_right = self.right_coords

            if pts_left is not None and pts_right is not None:
                pts_left = np.asarray(pts_left, dtype=np.float32)
                pts_right = np.asarray(pts_right, dtype=np.float32)

                pts_left_cv = pts_left[:, np.newaxis, :]
                pts_right_cv = pts_right[:, np.newaxis, :]

                # Undistort and rectify points
                pts_left_rect = cv2.undistortPoints(
                    pts_left_cv, self.K1, self.dist1, R=self.R1, P=self.P1
                )
                pts_right_rect = cv2.undistortPoints(
                    pts_right_cv, self.K2, self.dist2, R=self.R2, P=self.P2
                )

                # Flatten back to (N, 2)
                pts_left_rect = pts_left_rect.squeeze(1)
                pts_right_rect = pts_right_rect.squeeze(1)

                # Triangulate using your corrected coordinates
                points4D = cv2.triangulatePoints(
                    self.P1, self.P2, pts_left_rect.T, pts_right_rect.T
                )  # Produces output of size (X, Y, Z, W)
                points3D = (
                    (points4D[:3] / points4D[3]).T * 100
                )  # Transpose to get shape (N, 3) instead of (3, N) and multiply by 100 for cm
                points3D = np.squeeze(points3D)

                # Compute the raw projected points ema
                self.prev_points3D = self.computeEma(points3D)
                self.scatter1._offsets3d = (
                    self.prev_points3D[:, 0],
                    self.prev_points3D[:, 1],
                    self.prev_points3D[:, 2],
                )
                for line, (start, end) in zip(self.lines1, kp.HAND_SKELETON):
                    line.set_data(
                        [self.prev_points3D[start, 0], self.prev_points3D[end, 0]],
                        [self.prev_points3D[start, 1], self.prev_points3D[end, 1]],
                    )
                    line.set_3d_properties(
                        [self.prev_points3D[start, 2], self.prev_points3D[end, 2]]
                    )

                self.fig1.canvas.draw_idle()
                self.fig1.canvas.flush_events()

                feat, coords_proj, scale = self.feats(
                    torch.tensor(points3D),
                    torch.tensor(pts_left),
                    torch.tensor(pts_right),
                )
                feat = self.standardize(feat)

                with torch.inference_mode():
                    errors = self.model(feat, self.edge_index, self.batch)
                    errors = errors.view(1, 21, 3)

                    scale = torch.linalg.norm(
                        coords_proj[:, 9] - coords_proj[:, 0], dim=-1
                    )
                    pred_coords = coords_proj + (scale[:, None, None] * errors)
                    points3D_corr = pred_coords.squeeze(0).cpu().numpy()

                # Recenter optimized points at centroid of original raw projected hand palm keypoint coordinates
                points3D_center = points3D[self.PALM].mean(axis=0)
                points3D_corr_center = points3D_corr[self.PALM].mean(axis=0)
                translation = points3D_center - points3D_corr_center
                points3D_corr += translation

                # Compute the corrected points ema
                self.prev_points3D = self.computeEma(points3D_corr)

                self.scatter2._offsets3d = (
                    self.prev_points3D[:, 0],
                    self.prev_points3D[:, 1],
                    self.prev_points3D[:, 2],
                )
                for line, (start, end) in zip(self.lines2, kp.HAND_SKELETON):
                    line.set_data(
                        [self.prev_points3D[start, 0], self.prev_points3D[end, 0]],
                        [self.prev_points3D[start, 1], self.prev_points3D[end, 1]],
                    )
                    line.set_3d_properties(
                        [self.prev_points3D[start, 2], self.prev_points3D[end, 2]]
                    )

                self.fig2.canvas.draw_idle()
                self.fig2.canvas.flush_events()

    def start(self):
        take_frame_thread = Thread(target=self.take_frame, daemon=True)
        take_frame_thread.start()
        mp_thread = Thread(target=self.get_frame_mp)
        mp_thread.start()
        self.plot3D("mp")
        plt.show()

    def quit(self):
        self.running = False
        self.cam.release()
        cv2.destroyAllWindows()


left = Video(CAM)


def quit_streams():
    left.quit()
    listener.stop()


listener = keyboard.Listener(on_press=quit_streams)
listener.start()
left.start()
