import torch
import warnings
from mmengine.logging import MMLogger
from mmpose.apis import MMPoseInferencer
import time 

# hides python level warnings
warnings.filterwarnings("ignore")  

# tells the MM logger to only log actual errors and ignores the warnings
MMLogger.get_instance("mmengine", log_level = "ERROR")
MMLogger.get_instance("mmpose", log_level = "ERROR")
MMLogger.get_instance("mmdet", log_level = "ERROR")

class RTMPose():
    def __init__(self, frame_in, frame_out, dev = 'cpu'):
        self.inferencer = MMPoseInferencer('hand', device = dev)
        self.frame_in_path = frame_in
        self.frame_out_path = frame_out
        self.kp_coords = {}
    
    def get_keypoints(self):
        result_generator = self.inferencer(
            self.frame_in_path,
            return_vis = True,      # includes the visualized image in the output
            vis_out_dir = self.frame_out_path,     # saved visualized image to this path
            kpt_thr = 0.3,       # only takes keypoints with >= 30% confidence
            radius = 5,
            thickness = 2
        )
        
        return next(result_generator)
    
    def get_coords(self):
        result = self.get_keypoints()
        predictions = result['predictions'][0]
        for coords in predictions:
            keypoints = coords['keypoints']     # gives (x, y) coordinates of keypoints

            for kp_index, kp in enumerate(keypoints):
                self.kp_coords[kp_index] = kp
        
        return self.kp_coords
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose = RTMPose('/home/mrtcloud-1/Downloads/WIN_20260609_11_17_13_Pro.jpg', '/home/mrtcloud-1/Documents/Hand-Tracking-2/VideoTracking/frameout.jpeg', dev = device)

total_time = 0
for i in range(100):
    t0 = time.time()
    keypoints = pose.get_coords()
    t1 = time.time()
    total_time += (t1 - t0)

print(f"Average time per frame: {total_time / 100:.4f} seconds")