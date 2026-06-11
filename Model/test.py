import torch
import warnings
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from mmpose.apis import MMPoseInferencer
from mmengine.logging import MMLogger
import time

warnings.filterwarnings("ignore")
MMLogger.get_instance("mmengine", log_level="ERROR")
MMLogger.get_instance("mmpose", log_level="ERROR")
MMLogger.get_instance("mmdet", log_level="ERROR")

class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def infer(self, img):
        # preprocess
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # add batch dim

        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host']


class RTMPose:
    def __init__(self, frame_dir, engine_path='rtmpose_hand.trt', dev='cuda'):
        # TRT for pose
        self.trt_model = TRTModel(engine_path)
        # keep mmpose only for detection
        self.inferencer = MMPoseInferencer('hand', device=dev)
        self.detector = self.inferencer.inferencer.detector
        self.frame_dir = frame_dir
        self.frames = [
            cv2.imread(frame_dir + "FrameIn/left.jpeg"),
            cv2.imread(frame_dir + "FrameIn/right.jpeg"),
        ]

    def get_coords(self):
        img_coords = []
        for frame in self.frames:
            output = self.trt_model.infer(frame)
            coords = output.reshape(-1, 2)
            kp_coords = {i: coord.tolist() for i, coord in enumerate(coords)}
            img_coords.append(kp_coords)
        torch.cuda.empty_cache()  # add this
        return img_coords

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pose = RTMPose('/home/mrtcloud-1/Documents/Hand-Tracking-2/VideoTracking/')

total_time = []
for i in range(200):
    t0 = time.time()
    keypoints = pose.get_coords()
    t1 = time.time()
    total_time.append(t1-t0)
print(f"Average time per frame: {np.mean(total_time)} seconds")