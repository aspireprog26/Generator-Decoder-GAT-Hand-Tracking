import cv2
import mediapipe as mp
import numpy as np

"""
Use for RTMPose implementation:
import warnings
import pycuda.autoinit       
import tensorrt as trt
import pycuda.driver as cuda

Hide the non-critical warnings to keep console clean:
warnings.filterwarnings("ignore")
"""

# Define the Hand Skeleton connections. Each tuple draws a line between keypoints A and B
# The keypoints indices match the model's output ordering.

HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

ANGLE_JOINTS = [
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
]

"""
class TRTEngine:
    def __init__(self, engine):
        self.engine_path = engine

        self.logger = trt.Logger(trt.Logger.ERROR)      # Create TensorRT logger that only prints errors
        trt.init_libnvinfer_plugins(self.logger, "")   # Load any TensorRT plugins that are required by the engine

        # Deserialize the engine from the disk (rebuilds object from the TensorRT engine)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)      # Load the runtime TensorRT logger
            self.engine = runtime.deserialize_cuda_engine(f.read())     # deserialize into a TensorRT engine object to run inference
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}.")
        
        # Create execution context to run inference
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # Create cuda stream for asynchronous copies and inference
        self.stream = cuda.Stream()

        # Identify all input bindings (GPU memory address that holds incoming data for model input layer)
        self.input_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]

        # Identify all output bindings
        self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
  
        # Raise error if input length > 1 (this wrapper expects a single input tensor)
        if len(self.input_indices) != 1:
            raise RuntimeError(f"Expected exactly 1 input, found {len(self.input_indices)}.")
       
        self.input_idx = self.input_indices[0]      # Takes binding 0 as input since thats the image. Bindings 1 and 2 are scores and coords

    def infer(self, input_array: np.ndarray):   # Expects an N-dimensional array (Or input tensor)
        # TensorRT models expect BCHW input: Batch, Channels, Height, Width, raise error if lower dimension
        if input_array.ndim != 4: 
            raise ValueError(f"Expected BCHW input, got shape {input_array.shape}.")

        # Tell TensorRT the actual shape of the input if the engine uses dynamic input shapes
        if -1 in tuple(self.engine.get_binding_shape(self.input_idx)):
            self.context.set_binding_shape(self.input_idx, input_array.shape)
        
        bindings = [0] * self.engine.num_bindings       # create a list of zeros for every binding pointer in the TensorRT engine

        # Convert the input to the engine's expected dtype and ensure contiguous memory (Stores data and processes in an uninterrupted, sequential block of memory addresses)
        input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        input_array = np.ascontiguousarray(input_array.astype(input_dtype, copy = False))

        # Allocate GPU memory for the input and copy host to device
        d_input = cuda.mem_alloc(input_array.nbytes)
        cuda.memcpy_htod_async(d_input, input_array, self.stream)       # asynchronously copies CPU input to GPU input
        bindings[self.input_idx] = int(d_input)

        # Lists to keep track of output buffers on CPU and GPU
        cpu_outputs = []
        gpu_outputs = []
        output_shapes = []

        # Allocate memory for every output tensor
        for idx in self.output_indices:
            # Obtain the output shape from the execution context
            shape = tuple(self.context.get_binding_shape(idx))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Output shape still dynamic for binding {idx}: {shape}.")
            
            # Determine output dtype and total number of elements
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            size = int(np.prod(shape))

            # Create CPU memory buffer for TensorRT to recieve output data copied back from the GPU
            cpu_mem = cuda.pagelocked_empty(size, dtype)
            gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
            
            bindings[idx] = int(gpu_mem)
            cpu_outputs.append(cpu_mem)
            gpu_outputs.append(gpu_mem)
            output_shapes.append(shape)
        
        # Run the engine asynchronously on the CUDA stream
        self.context.execute_async_v2(bindings = bindings, stream_handle = self.stream.handle)

        # COpy each output back from gpu to cpu
        for cpu_mem, gpu_mem in zip(cpu_outputs, gpu_outputs):
            cuda.memcpy_dtoh_async(cpu_mem, gpu_mem, self.stream)
        
        # Wait until all queued CUDA work has finished
        self.stream.synchronize()

        # Reshape flat outputs into their original tensor shape
        outputs = [cpu.reshape(shape) for cpu, shape in zip(cpu_outputs, output_shapes)]
        return outputs

class RTMPose:
    def __init__(self, engine):
        self.engine = TRTEngine(engine)
        self.input_w, self.input_h = (256, 256)
        self.conf = 0.12

    def read_image(self, image_array):
        # Convert array input to NumPy array
        img = np.asarray(image_array)
        if img.ndim != 3:
            raise ValueError(f"Expected image with 3-dims, instead got {img.shape}.")
        
        return img
    
    def preprocess(self, img_bgr: np.ndarray):
        # Normalize image array used by RTM vision backbone
        mean = np.array([123.675, 116.28, 103.53], dtype = np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype = np.float32)


        # load image as RGB from BGR
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h), interpolation = cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = (img - mean) / std

        # Convert from HWC to CHW because TensorRT expect (Channels, Height, Width)
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def decode_outputs(self, outputs):
        
       # RTMPose exports take different output forms depending on model export
       # Supported Forms:
       # 1) One Output: (B, K, 2) or (B, K, 3) -> Direct Coordinates
       # 2) One Output: (B, K, H, W) -> Heatmaps
       # 3) Two Outputs: (B, K, L) -> SimCC-style x,y distributions


        if len(outputs) == 1:
            out = outputs[0]

            # Direct coordinate output: last dimension contains x, y and confidence
            if out.ndim == 3 and out.shape[-1] in (2, 3):
                coords = out[..., :2].astype(np.float32)
                if out.shape[-1] == 3:
                    scores = out[..., 2].astype(np.float32)
                else:
                    scores = np.ones(coords.shape[:2], dtype = np.float32)
                return coords, scores
            
            # Heatmap output: Find the maximum probability pixel for each keypoint
            if out.ndim == 4:
                b, k, h, w = out.shape
                coords = np.zeros((b, k, 2), dtype = np.float32)
                scores = np.zeros((b, k), dtype = np.float32)

                for bi in range(b):
                    for ki in range(k):
                        heatmap = out[bi, ki]
                        pos = int(np.argmax(heatmap))
                        y, x = divmod(pos, w)

                        # Convert heatmap coordinates to input-image coordinates
                        coords[bi, ki, 0] = x * (self.input_w / max(w - 1, 1))
                        coords[bi, ki, 1] = y * (self.input_h / max(h - 1, 1))
                        scores[bi, ki] = float(heatmap[y, x])
                
                return coords, scores
            
        # SimCC-style output distributions for x and y
        if len(outputs) >= 2:
            x_out, y_out = outputs[0], outputs[1]

            if x_out.ndim == 3 and y_out.ndim == 3 and y_out.shape[:2]:
                b, k, lx = x_out.shape
                _, _, ly = y_out.shape
                coords = np.zeros((b, k, 2), dtype = np.float32)
                scores = np.zeros((b, k), dtype = np.float32)

                for bi in range(b):
                    for ki in range(k):
                        x_pos = int(np.argmax(x_out[bi, ki]))
                        y_pos = int(np.argmax(y_out[bi, ki]))
                        
                        # Convert distribution indixes into input-image coordinates
                        coords[bi, ki, 0] = x_pos * (self.input_w / max(lx - 1, 1))
                        coords[bi, ki, 1] = y_pos * (self.input_h / max(ly - 1, 1))

                        # Approximate confidence by combining best x and y probabilities
                        scores[bi, ki] = float(x_out[bi, ki, x_pos] * y_out[bi, ki, y_pos])
                
                return coords, scores

    def orig_scale(self, coords, orig_w, orig_h):
        # Make a float copy so we can rescale coordinates without modifying the original array
        scaled = coords.copy().astype(np.float32)

        # Rescales x and y coordinates to original size 
        scaled[..., 0] = scaled[..., 0] * (orig_w / float(self.input_w))
        scaled[..., 1] = scaled[..., 1] * (orig_h / float(self.input_h))

        return scaled   

    def draw_hand(self, img_bgr, coords, scores):
        vis = img_bgr.copy()
        # Draw the bones first so the keypoint circles appear on top
        for a, b in HAND_SKELETON:
            if a in coords and b in coords:
                # Skip low confidence keypoints
                if scores[a] < self.conf or scores[b] < self.conf:
                    continue
            
                p1 = coords[a]
                p2 = coords[b]

                pt1 = (int(round(p1[0])), int(round(p1[1])))
                pt2 = (int(round(p2[0])), int(round(p2[1])))
                cv2.line(vis, pt1, pt2, (12, 27, 196), 2)      # Color the lines of the keypoint skeleton

        # Draw each keypoint as filled circle with border
        for idx, pt in coords.items():
            if scores[idx] < self.conf:
                continue

            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(vis, center, 5, (0, 0, 255), -1)         # Inner circle
            cv2.circle(vis, center, 5, (0, 0, 0), 1)      # Border circle  
        
        return vis
            
    def get_keypoints(self, left_frame, right_frame):
        results = []
        score = []

        image_list = [left_frame, right_frame]

        for images in image_list:
            image = self.read_image(images)

            # Preprocess and add batch dimension
            batch = np.expand_dims(self.preprocess(image), axis = 0)
            outputs = self.engine.infer(batch)

            coords, scores = self.decode_outputs(outputs)
            
            # Remove exports with additional batch dimension
            if coords.ndim == 3:
                coords = coords[0]
                scores = scores[0]
            
            # Scale coordinates back to original image size
            h, w = image.shape[:2]      # Only takes height and width, ignores RGB channel
            scaled_coords = self.orig_scale(coords, w, h)

            kp_coords = {}
            for i in range(scaled_coords.shape[0]):
                kp_coords[i] = scaled_coords[i].tolist()

            results.append(kp_coords)
            score.append(scores)

        return results, score 
"""


class MediaPipe:
    def __init__(self, model_complexity=1, detection_scale=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detection_scale = detection_scale

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=0.3,
        )

    def get_keypoints(self, frame):
        kps = np.zeros((21, 2), dtype=np.float32)
        handedness_label = None
        handedness_score = 0.0

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.detection_scale != 1.0:
            rgb_frame = cv2.resize(
                rgb_frame,
                (
                    max(1, int(w * self.detection_scale)),
                    max(1, int(h * self.detection_scale)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    kps[idx] = [px, py]

                handedness_label = hand_info.classification[0].label

                # If the image is not selfie view / mirrored
                handedness_label = "Left" if handedness_label == "Right" else "Right"
                handedness_score = hand_info.classification[0].score

        return kps, handedness_label, handedness_score

    def draw_hand(self, coords, frame):
        frame = frame.copy()
        for a, b in HAND_SKELETON:
            p1 = coords[a]
            p2 = coords[b]

            pt1 = (round(p1[0]), round(p1[1]))
            pt2 = (round(p2[0]), round(p2[1]))
            cv2.line(
                frame, pt1, pt2, (12, 27, 196), 5
            )  # Color the lines of the keypoint skeleton

        for i in range(21):
            x = coords[i][0]
            y = coords[i][1]
            center = (round(x), round(y))
            cv2.circle(frame, center, 7, (0, 0, 255), -1)  # Inner circle
            cv2.circle(frame, center, 7, (0, 0, 0), 1)  # Border circle

        return frame
