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
    def __init__(self, frame_dir, dev = 'cpu'):
        self.inferencer = MMPoseInferencer('hand', device = dev)
        self.frame_dir = frame_dir
    
    def get_keypoints(self):
        result_generator = self.inferencer(
            [self.frame_dir + "FrameIn/left.jpeg", self.frame_dir + "FrameIn/right.jpeg"],
            return_vis = True,      # includes the visualized image in the output
            vis_out_dir = self.frame_dir + "FrameOut",     # saved visualized image to this path
            kpt_thr = 0.3,       # only takes keypoints with >= 30% confidence
            radius = 5,
            thickness = 2,
            batch_size = 2
        )
        
        # return all results in the generator
        results = [] 
        for result in result_generator:
            results.append(result)

        return results
    
    def get_coords(self):
        results = self.get_keypoints()
        img_coords = []
        
        for result in results:      # get each generator result
            for predictions in result['predictions']:       # get each prediction in the batch per image
                kp_coords = {}
                for coords in predictions:      # obtain the coordinates per prediction              
                    for kp_index, coord in enumerate(coords['keypoints']):      # obtain the index and coordinate for each keypoint
                        kp_coords[kp_index] = coord
                img_coords.append(kp_coords)
    
        return img_coords

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose = RTMPose('/home/mrtcloud-1/Documents/Hand-Tracking-2/VideoTracking/', dev = device)
#keypoints = pose.get_coords()

#do warmup
for _ in range(50):
    keypoints = pose.get_coords()

"""
for i in range(200):
    t0 = time.time()
    keypoints = pose.get_coords()
    t1 = time.time()
    print(t1 - t0)
"""

keypoints = pose.get_coords()
print(keypoints)