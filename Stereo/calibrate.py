import cv2
import numpy as np 
from pathlib import Path

PATTERN_SIZE = (8, 5)   # Size of checkerboard inner corners
SQUARE_SIZE = 0.038     # In meters
DIR = r"C:\Users\Test\Documents\StereoImages\Stereo2"

def calibrate():
    dir = Path(DIR)
    fs = cv2.FileStorage(r"C:\Users\Test\Documents\Hand-Tracking\Stereo\stereo.yml", cv2.FILE_STORAGE_WRITE)
    img_size = None
    
    objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)       # Create an empty array to store 3D points of shape (num_corners, 3) with each row (X, Y, Z)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)                       # fill in the X and Y coordinates for each chessboard corner        
    objp *= SQUARE_SIZE

    obj_points = []     # 3D points in the real world space
    left_points = []    # 2D points in left image
    right_points = []   # 2D points in right image

    for img in dir.iterdir():
        image = cv2.imread(str(img))
        h, w = image.shape[:2]
        half = w // 2

        left  = image[:, :half]
        right = image[:, half:]

        grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, PATTERN_SIZE)       
        retR, cornersR = cv2.findChessboardCorners(grayR, PATTERN_SIZE)

        if retL and retR:
            obj_points.append(objp)

            # Find the left and right chessboard corners with (11, 11) search window and (-1, -1) ignored region to use full window. 
            # Set criteria max iterations to 30 and early stop to 0.001 if corner moves less than that pixel difference for refinement control.
            cornersL = cv2.cornerSubPix(
                grayL, cornersL, (11, 11), (-1, -1),
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            cornersR = cv2.cornerSubPix(
                grayR, cornersR, (11, 11), (-1, -1),
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            left_points.append(cornersL)
            right_points.append(cornersR)

        img_size = grayL.shape[::-1]        # Returns image size in form (width, height)

    # Obtain reprojection error, intrinsic camera matrix, distortion coefficients, rotation vectors, and translation vectors for each camera
    retL, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, left_points, img_size, None, None)
    retR, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points, right_points, img_size, None, None)

    # Obtain rotation matrix, translation vector, essential matrix (undistorted image coordinates), and fundamental matrix to obtain epipolar lines from left to right camera.
    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        left_points,
        right_points,
        K1, dist1,
        K2, dist2,
        img_size, 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags = cv2.CALIB_FIX_INTRINSIC     # Tells OpenCV to not change the intrinsic parameters (K1, K2, dist1, dist2) since they're already calibrated
    )

    # Warp both images so they appear as if they were perfectly aligned like before 
    # Obtain rectification rotation matrices, projection matrices, disparity to depth mapping matrix, and regions of interest (areas containing reliable pixels) for left and right images
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1,
        K2, dist2,
        img_size,
        R, T,
        flags = cv2.CALIB_ZERO_DISPARITY,       # Tells OpenCV to make principal points of rectified image line up as closely as possible
        alpha = 0,                              # Crop away invalid regions so no black borders after distortion of images
    )

    print(retL)
    print(retR)
    print(ret)

    fs.write("K1", K1)
    fs.write("K2", K2)
    fs.write("dist1", dist1)
    fs.write("dist2", dist2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.write("IMG_SIZE", img_size)

    fs.release()
    print("Calibration Complete.")

calibrate()