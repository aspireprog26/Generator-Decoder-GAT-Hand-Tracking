# Hand-Tracking-2

<img width="800" height="600" alt="1" src="https://github.com/user-attachments/assets/b0345c52-477e-4a5b-af94-8e876ccc55ca" />

This project computes real-time 3D tracking of hand pose, with a stereo camera and RTMPose. The model was accelerated through TensorRT and CUDA for the RTMPose engine to enable a batch inference time on the left and right stereo images of up to 120 FPS. The stereo camera was calibrated via a (8, 5) chessboard (inner corners), and OpenCV. 

We utilized the STB Dataset to obtain anatomically constrained 3D points in the real world coordinate space, and optimized the local objective of angle, bone length ratios, and keypoint ordering per finger to obtain anatomically and geographically accurate points. 
