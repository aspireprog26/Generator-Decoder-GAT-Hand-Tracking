import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Keypoints")
from keypointdetection import MediaPipe
from constrain import OptimizeHands

# Define the Hand Skeleton connections. Each tuple draws a line between keypoints A and B
HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

if __name__ == "__main__":
    pose = MediaPipe()
    image = cv2.imread(r"/home/mrtcloud-1/Documents/StereoDataset/Noisy/0542.jpg")
    h, w = image.shape[:2]
    half = w // 2

    left = image[:, :half]
    right = image[:, half:]
    frames = [left, right]

    left_kps = pose.get_keypoints(left)
    right_kps = pose.get_keypoints(right)
    kps = [left_kps, right_kps]

    for i in range(2):
        out = Path(r"/home/mrtcloud-1/Documents/Hand-Tracking-2/FrameOut") / f"{i}.jpg"
        vis = pose.draw_hand(kps[i], frames[i])
        cv2.imwrite(out, vis)

    fs = cv2.FileStorage(
        r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Stereo/stereo.yml",
        cv2.FILE_STORAGE_READ,
    )

    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    K1 = fs.getNode("K1").mat()
    K2 = fs.getNode("K2").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    dist1 = fs.getNode("dist1").mat()
    dist2 = fs.getNode("dist2").mat()
    fs.release()

    pts_left = np.asarray(left_kps, dtype=np.float32).reshape(-1, 1, 2)
    pts_right = np.asarray(right_kps, dtype=np.float32).reshape(-1, 1, 2)

    pts_left_rect = cv2.undistortPoints(pts_left, K1, dist1, R=R1, P=P1)
    pts_right_rect = cv2.undistortPoints(pts_right, K2, dist2, R=R2, P=P2)

    # Flatten back to (N, 2)
    pts_left_rect = pts_left_rect.squeeze(1)
    pts_right_rect = pts_right_rect.squeeze(1)

    # Obtain 4D points and scale to 3D
    points4D = cv2.triangulatePoints(P1, P2, pts_left_rect.T, pts_right_rect.T)
    points3D = (points4D[:3] / points4D[3]).T * 100
    points3D = np.squeeze(points3D)

    #   hand_optimizer = OptimizeHands(points3D, left, right)
    # optimized_kps = hand_optimizer.optimize()

    # points3D = (optimized_kps[0] + optimized_kps[1]) / 2
    points3D = np.load("/home/mrtcloud-1/Documents/StereoDataset/Clean/0000.npy")[1]
    # points3D = (optimized_kps[2] + optimized_kps[2]) / 2
    print(points3D)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.zaxis.set_inverted(True)
    ax.view_init(elev=220, azim=130, roll=0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(
        points3D[:, 0],
        points3D[:, 1],
        points3D[:, 2],
        color=(196 / 255, 12 / 255, 27 / 255),
        s=15,
    )
    for start, end in HAND_SKELETON:
        ax.plot(
            [points3D[start, 0], points3D[end, 0]],
            [points3D[start, 1], points3D[end, 1]],
            [points3D[start, 2], points3D[end, 2]],
            "b-",
        )
    plt.title("3D Mapped Hand Skeleton Keypoints (In Centimeters)")
    plt.show()
