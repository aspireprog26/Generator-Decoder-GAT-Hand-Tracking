import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# FPS variables
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # MediaPipe wants RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Measure MediaPipe time
    mp_start = time.time()

    results = hands.process(rgb)

    mp_end = time.time()

    mp_time = (mp_end - mp_start) * 1000  # ms

    # Calculate FPS
    current_time = time.time()
    current_fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Smooth FPS
    fps = 0.9 * fps + 0.1 * current_fps

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display FPS and inference time
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"MediaPipe: {mp_time:.1f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "MediaPipe Hand Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()