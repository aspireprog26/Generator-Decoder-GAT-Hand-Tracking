import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from manotrainer import Trainer as MANOTrainer
from matplotlib import animation
from model import AnatomyModel, MANOModel
from posttrain import Loss
from pretrainer import Trainer as PreTrainer
from torch_geometric.data import Batch
from utils import Stats, procrustesAlign, stereoTransform

sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Keypoints")
sys.path.insert(0, "/home/miket/Documents/Hand-Tracking-2/Dataset")

from handedgeindex import hand_edge_index  # type: ignore
from keypointdetection import HAND_SKELETON, MediaPipe  # type: ignore

sample_eval = True

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
STEREO_CALIB_PATH = "/home/miket/Documents/Hand-Tracking-2/Stereo/stereo.yml"
EMA_ALPHA = 0.6

with open("/home/miket/Documents/Hand-Tracking-2/Model/decpostconfigs.json", "r") as f:
    dec_post_configs = json.load(f)

with open("/home/miket/Documents/Hand-Tracking-2/Model/manoconfigs.json", "r") as f:
    mano_configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

joint_predictor = AnatomyModel(
    dec_post_configs["input_size"],
    dec_post_configs["decoder_hidden_size"],
    dec_post_configs["decoder_hidden1"],
    dec_post_configs["decoder_output_size"],
    dec_post_configs["decoder_dropout"],
    generator=False,
).to(device)

joint_weights = torch.load(
    (Path(dec_post_configs["model_dir"]) / f"{dec_post_configs['decoder_model_name']}"),
    weights_only=True,
    map_location=device,
)
joint_predictor.load_state_dict(joint_weights)
joint_predictor.eval()

mano_model = MANOModel(
    mano_configs["input_size"],
    mano_configs["hidden_size"],
    mano_configs["hidden1"],
    mano_configs["dropout"],
    mano_configs["mano_root"],
    mano_configs["ncomps"],
).to(device)

mano_weights = torch.load(
    (Path(mano_configs["model_dir"]) / f"{mano_configs['model_name']}"),
    weights_only=True,
    map_location=device,
)
mano_model.load_state_dict(mano_weights)
mano_model.eval()

standardize = Stats(device).standardize
feats = PreTrainer(
    dec_post_configs,
    None,
    None,
    joint_predictor,
    joint_predictor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
).features

mano_feats = MANOTrainer(
    mano_model, joint_predictor, mano_configs, None, None, None, None, None, device
).getNewFeatures

criterion = Loss(
    dec_post_configs["delta1"],
    dec_post_configs["delta2"],
    dec_post_configs["w1"],
    dec_post_configs["w2"],
    dec_post_configs["w3"],
    dec_post_configs["w4"],
).criterion


def collate(batch):
    graphs, targets, coords_proj = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    coords_proj = torch.stack(coords_proj, dim=0).float()
    return (batch, targets, coords_proj)


def plot(ax, points3D, orig=True):
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

    ax.set_title(
        "Raw 3D Projected Stereo Mapped Hand Keypoints"
        if orig
        else "Corrected 3D Projected Stereo Mapped Hand Keypoints"
    )


def setupAxis(ax, title, point_color, line_style):
    ax.zaxis.set_inverted(True)
    ax.view_init(elev=220, azim=130, roll=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    scatter = ax.scatter([], [], [], color=point_color, s=15, clip_on=True)
    lines = []
    for _ in HAND_SKELETON:
        (line,) = ax.plot([], [], [], line_style, clip_on=True)
        lines.append(line)

    return scatter, lines


def rescaleAxis(ax, points3D, pad_ratio=0.15, min_pad=5.0):
    mins = points3D.min(axis=0)
    maxs = points3D.max(axis=0)
    pads = np.maximum((maxs - mins) * pad_ratio, min_pad)
    ax.set_xlim3d(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim3d(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim3d(mins[2] - pads[2], maxs[2] + pads[2])


def updatePoints(ax, scatter, lines, points3D):
    scatter._offsets3d = (points3D[:, 0], points3D[:, 1], points3D[:, 2])
    for line, (start, end) in zip(lines, HAND_SKELETON):
        line.set_data(
            [points3D[start, 0], points3D[end, 0]],
            [points3D[start, 1], points3D[end, 1]],
        )
        line.set_3d_properties([points3D[start, 2], points3D[end, 2]])
    rescaleAxis(ax, points3D)


def animatePlot(frames_video, frames_orig, frames_corr, fps=30):
    """Animate the source video alongside the raw/corrected 3D keypoint
    plots, all three panes advancing together at the source frame rate."""
    fig = plt.figure(figsize=(18, 6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    ax0.axis("off")
    ax0.set_title("Input Video")
    img_artist = ax0.imshow(cv2.cvtColor(frames_video[0], cv2.COLOR_BGR2RGB))

    scatter1, lines1 = setupAxis(
        ax1,
        "Raw 3D Projected Hand Keypoints",
        (196 / 255, 12 / 255, 27 / 255),
        "blue",
    )
    scatter2, lines2 = setupAxis(
        ax2,
        "Corrected 3D Projected Hand Keypoints",
        (2 / 255, 108 / 255, 87 / 255),
        "teal",
    )

    def update(i):
        img_artist.set_data(cv2.cvtColor(frames_video[i], cv2.COLOR_BGR2RGB))
        updatePoints(ax1, scatter1, lines1, frames_orig[i])
        updatePoints(ax2, scatter2, lines2, frames_corr[i])
        return (img_artist, scatter1, scatter2, *lines1, *lines2)

    # interval is in ms; matches the source video's frame rate so playback
    # speed lines up with the original clip
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_video),
        interval=max(1000 / fps, 1),
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()
    return anim


def ema(arr, alpha, prev):
    return arr * alpha + (1 - alpha) * prev


def loadStereoCalibration(calib_path=STEREO_CALIB_PATH):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)

    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    K1 = fs.getNode("K1").mat()
    K2 = fs.getNode("K2").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    dist1 = fs.getNode("dist1").mat()
    dist2 = fs.getNode("dist2").mat()
    fs.release()

    return P1, P2, K1, K2, R1, R2, dist1, dist2


def combineFrames(left_frame, right_frame):
    h1, w1 = left_frame.shape[:2]
    h2, w2 = right_frame.shape[:2]

    if h1 != h2:
        target_h = min(h1, h2)
        left_frame = cv2.resize(left_frame, (int(w1 * target_h / h1), target_h))
        right_frame = cv2.resize(right_frame, (int(w2 * target_h / h2), target_h))

    return np.hstack((left_frame, right_frame))


def processFrame(frame, pose_left, pose_right, calib, executor):
    P1, P2, K1, K2, R1, R2, dist1, dist2 = calib

    _, w = frame.shape[:2]
    half = w // 2

    left = frame[:, :half]
    right = frame[:, half:]

    # Run left/right keypoint detection concurrently. Each half uses its
    # own MediaPipe instance (pose_left/pose_right) -- MediaPipe's Hands
    # solution keeps internal tracking state between calls and isn't
    # thread-safe, so sharing one instance across threads risks corrupting
    # that state. Two independent instances make this safe.
    left_future = executor.submit(pose_left.get_keypoints, left)
    right_future = executor.submit(pose_right.get_keypoints, right)
    left_kps, _, _ = left_future.result()
    right_kps, _, _ = right_future.result()

    left_annotated = pose_left.draw_hand(left_kps, left)
    right_annotated = pose_right.draw_hand(right_kps, right)
    annotated_frame = combineFrames(left_annotated, right_annotated)

    pts_left = np.asarray(left_kps, dtype=np.float32).reshape(-1, 1, 2)
    pts_right = np.asarray(right_kps, dtype=np.float32).reshape(-1, 1, 2)

    pts_left_rect = cv2.undistortPoints(pts_left, K1, dist1, R=R1, P=P1)
    pts_right_rect = cv2.undistortPoints(pts_right, K2, dist2, R=R2, P=P2)

    pts_left_rect = pts_left_rect.squeeze(1)
    pts_right_rect = pts_right_rect.squeeze(1)

    points4D = cv2.triangulatePoints(P1, P2, pts_left_rect.T, pts_right_rect.T)
    points3D = (points4D[:3] / points4D[3]).T * 100
    points3D_orig = np.squeeze(points3D)

    points3D_t = torch.tensor(points3D_orig).unsqueeze(0).to(device)
    left_kps_t = torch.tensor(left_kps).unsqueeze(0).to(device)
    right_kps_t = torch.tensor(right_kps).unsqueeze(0).to(device)

    raw_feat, coords_proj, _ = feats(points3D_t, left_kps_t, right_kps_t)
    raw_feat = raw_feat.reshape(1, 21, 19)
    feat = standardize(raw_feat)

    edge_index = hand_edge_index.to(device)
    batch = torch.zeros(21, dtype=torch.long, device=device)

    with torch.inference_mode():
        joint_pred_coords = joint_predictor(feat, edge_index, batch)
        updated_features = mano_feats(raw_feat, joint_pred_coords)
        updated_features = standardize(updated_features, decoder=False)
        pred_coords = mano_model(updated_features, edge_index, batch)
        pred_coords = procrustesAlign(pred_coords, joint_pred_coords)
        pred_coords = stereoTransform(pred_coords, coords_proj)[0]
        points3D_corr = pred_coords.squeeze(0).cpu().numpy()

    return points3D_orig, points3D_corr, annotated_frame


def evalVideo(sample: str, alpha: float = EMA_ALPHA):
    # model_complexity=0 uses MediaPipe's lighter hand model. Separate
    # instances so left/right detection can run concurrently on their own
    # threads without sharing internal tracking state.

    pose_left = MediaPipe(model_complexity=0)
    pose_right = MediaPipe(model_complexity=0)
    calib = loadStereoCalibration()

    cap = cv2.VideoCapture(sample)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {sample}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    frames_video = []
    frames_orig = []
    frames_corr = []
    frame_times = []

    prev_raw_points3D = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            points3D_orig, points3D_corr, annotated_frame = processFrame(
                frame, pose_left, pose_right, calib, executor
            )
            t1 = time.time()

            if prev_raw_points3D is None:
                raw_display_points3D = points3D_orig.copy()
            else:
                raw_display_points3D = ema(points3D_orig, alpha, prev_raw_points3D)
            prev_raw_points3D = raw_display_points3D

            frames_video.append(annotated_frame)
            frames_orig.append(raw_display_points3D)
            frames_corr.append(points3D_corr)
            frame_times.append(t1 - t0)
    cap.release()

    if not frames_orig:
        raise ValueError(f"No frames could be read from video: {sample}")

    avg_time = sum(frame_times) / len(frame_times)

    animatePlot(frames_video, frames_orig, frames_corr, fps=fps)

    return avg_time


def evalModel(sample, alpha: float = EMA_ALPHA):
    ext = Path(sample).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return evalVideo(sample, alpha=alpha)


avg_time = evalModel("/home/miket/Documents/StereoDataset/Video/2.mp4")
print(f"Average Time per Frame: {avg_time: .4f}")
