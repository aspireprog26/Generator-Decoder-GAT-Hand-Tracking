from mmpose.apis import MMPoseInferencer
import torch
import cv2

class RTMPose():
    def __init__(self, frame_in, frame_out, dev = 'cpu'):
        self.inferencer = MMPoseInferencer('hand', device = dev)
        self.frame_in_path = frame_in
        self.frame_out_path = frame_out
    
    def generate_result(self):
        result_generator = self.inferencer(
            self.frame_in_path,
            return_vis = True,      # includes the visualized image in the output
            vis_out_dir = self.frame_out_path,     # saved visualized image to this path
            kpt_thr = 0.3       # only takes keypoints with >= 30% confidence
        )

        return next(result_generator)
    
    def get_keypoints(self):
        result = self.generate_result()
        predictions = result['predictions'][0]
        for coords in predictions:
            keypoints = coords['keypoints']     # gives (x, y) coordinates of keypoints
            scores = coords['keypoint_scores']

            for kp_index, (kp, score) in enumerate(zip(keypoints, scores)):
                print(f"Keypoint {kp_index}: (x = {kp[0]:.1f}, y = {kp[1]:.1f}), score = {score:.2f}")
            print("\n")
            
        visualized_img = result['visualization']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose_engine = RTMPose('/home/mrtcloud-1/Documents/IMG_0775.jpeg', '/home/mrtcloud-1/Documents/Hand-Tracking-2/VideoTracking/frameout.jpg', dev = device)
pose_engine.get_keypoints()