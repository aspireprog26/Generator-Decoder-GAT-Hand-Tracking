"""
3D Hand Keypoints Detection with MediaPipe
- Captures webcam feed
- Detects 21 hand landmarks in 3D (x, y, z)
- Plots them live in a 3D matplotlib figure
- Prints coordinates to the console

Requirements:
    pip install mediapipe opencv-python matplotlib numpy
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark names for printing
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Connections between landmarks for 3D plot lines
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)


def print_landmarks(hand_label: str, landmarks_3d: list[tuple]) -> None:
    """Print all 21 keypoints for a detected hand."""
    print(f"\n{'─'*50}")
    print(f"  Hand: {hand_label}")
    print(f"{'─'*50}")
    print(f"  {'#':<4} {'Landmark':<25} {'X':>8} {'Y':>8} {'Z':>8}")
    print(f"  {'─'*53}")
    for i, (x, y, z) in enumerate(landmarks_3d):
        print(f"  {i:<4} {LANDMARK_NAMES[i]:<25} {x:>8.4f} {y:>8.4f} {z:>8.4f}")


def setup_3d_axes(ax, title: str = "3D Hand Keypoints") -> None:
    """Configure a 3D matplotlib axis."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z (depth)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.2, 0.2)
    ax.invert_yaxis()   # Match image coordinate system
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(labelsize=7)


def draw_hand_3d(ax, landmarks_3d: list[tuple], color: str = "#00e5ff") -> None:
    """Draw landmark points and skeleton connections on a 3D axis."""
    xs, ys, zs = zip(*landmarks_3d)

    # Draw bones / connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        sx, sy, sz = landmarks_3d[start_idx]
        ex, ey, ez = landmarks_3d[end_idx]
        ax.plot([sx, ex], [sy, ey], [sz, ez],
                color=color, linewidth=1.5, alpha=0.7)

    # Draw joint points
    ax.scatter(xs, ys, zs, c=color, s=40, zorder=5, alpha=0.95,
               edgecolors="white", linewidths=0.4)

    # Highlight fingertips (indices 4, 8, 12, 16, 20)
    tips = [4, 8, 12, 16, 20]
    tip_coords = [landmarks_3d[i] for i in tips]
    tx, ty, tz = zip(*tip_coords)
    ax.scatter(tx, ty, tz, c="#ff4081", s=80, zorder=6,
               edgecolors="white", linewidths=0.6)

    # Label wrist
    wx, wy, wz = landmarks_3d[0]
    ax.text(wx, wy, wz, " WRIST", fontsize=6, color="white", alpha=0.8)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam. Check your camera index.")
        return

    # ── Matplotlib: interactive 3D figure ───────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(12, 5), facecolor="#0f0f1a")
    fig.suptitle("MediaPipe – Live 3D Hand Keypoints", color="white",
                 fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(121, projection="3d")  # Left hand / first detected
    ax2 = fig.add_subplot(122, projection="3d")  # Right hand / second detected
    setup_3d_axes(ax1, "Hand 1")
    setup_3d_axes(ax2, "Hand 2")

    plt.tight_layout()

    print("✅  Hand keypoints tracker started.")
    print("   • Press  Q  in the webcam window to quit.")
    print("   • 3D plot updates live.")
    print("   • Coordinates are printed each frame a hand is detected.\n")

    frame_count = 0
    PRINT_EVERY_N_FRAMES = 15   # Reduce console spam; set to 1 for every frame

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to grab frame.")
                break

            frame_count += 1

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw 2D landmarks on the webcam preview
            annotated = frame.copy()
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            cv2.imshow("MediaPipe Hands – press Q to quit", annotated)

            # ── Update 3D plot ───────────────────────────────────────────────
            if results.multi_hand_landmarks and results.multi_handedness:
                ax1.cla()
                ax2.cla()
                setup_3d_axes(ax1, "Hand 1")
                setup_3d_axes(ax2, "Hand 2")

                for hand_idx, (hand_lms, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    label = handedness.classification[0].label  # "Left" / "Right"
                    score = handedness.classification[0].score

                    # Extract (x, y, z) for all 21 landmarks
                    landmarks_3d = [
                        (lm.x, lm.y, lm.z) for lm in hand_lms.landmark
                    ]

                    # Choose axis and color per hand slot
                    ax = ax1 if hand_idx == 0 else ax2
                    color = "#00e5ff" if label == "Right" else "#69ff47"
                    ax.set_title(f"{label} Hand  (conf {score:.2f})",
                                 fontsize=11, fontweight="bold",
                                 color=color, pad=8)
                    draw_hand_3d(ax, landmarks_3d, color=color)

                    # Print to console
                    if frame_count % PRINT_EVERY_N_FRAMES == 0:
                        print_landmarks(f"{label} (hand #{hand_idx + 1})", landmarks_3d)

                fig.canvas.draw()
                fig.canvas.flush_events()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n👋  Quit requested.")
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()