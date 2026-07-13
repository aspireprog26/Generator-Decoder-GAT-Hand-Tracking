import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
# 1. Setup
mp_hands = mp.solutions.hands
# static_image_mode=True is optimized for processing separate, unrelated images
hands = mp_hands.Hands(static_image_mode = True, max_num_hands = 1, min_detection_confidence = 0.2)

# 2. Load and process image
image = cv2.imread(r'c:\Users\Test\Downloads\IMG_0776.jpeg  ')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

# 3. Extract to NumPy array
if results.multi_hand_landmarks:
    # Get the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Create an empty (21, 3) array
    landmarks_array = np.zeros((21, 3))
    
    # Fill the array
    for i, lm in enumerate(hand_landmarks.landmark):
        landmarks_array[i] = [lm.x, lm.y, lm.z]
        
    print("NumPy Array Shape:", landmarks_array.shape)
    print("Coordinates of Wrist (Index 0):\n", landmarks_array)
else:
    print("No hand detected.")

hands.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 1. Plot the Landmarks (Points)
# landmarks_array[:,0] is all x values, [:,1] is y, [:,2] is z
ax.scatter(landmarks_array[:, 0], landmarks_array[:, 1], landmarks_array[:, 2], c='red')

# 2. Plot the Connections (Lines)
mp_hands = mp.solutions.hands
for connection in mp_hands.HAND_CONNECTIONS:
    start_idx = connection[0]
    end_idx = connection[1]
    
    # Extract the two points for this bone
    p1 = landmarks_array[start_idx]
    p2 = landmarks_array[end_idx]
    
    # Plot the line between them
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='blue')

# Optional: Set labels and flip the Y-axis if needed to match image orientation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.invert_yaxis() # In image processing, Y increases downwards

plt.show()