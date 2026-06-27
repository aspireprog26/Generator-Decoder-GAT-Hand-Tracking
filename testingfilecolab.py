import cv2
import warnings 
import numpy as np
import pycuda.autoinit      # Initializes cuda when imported
import tensorrt as trt
from pathlib import Path
import pycuda.driver as cuda
import matplotlib.pyplot as plt

# Hide the non-critical warnings to keep console clean
warnings.filterwarnings("ignore")

# Define the Hand Skeleton connections. Each tuple draws a line between keypoints A and B
# The keypoints indices match the model's output ordering.

HAND_SKELETON = [ 
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

class TRTEngine:
    def __init__(self, engine):
        self.engine_path = engine

        self.logger = trt.Logger(trt.Logger.ERROR)      # Create TensorRT logger that only prints errors
        trt.init_libnvinfer_plugins(self.logger, "")   # Load any TensorRT plugins that are required by the engine

        # Deserialize the engine from the disk
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)      
            self.engine = runtime.deserialize_cuda_engine(f.read())     
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}.")
        
        # Create execution context to run inference
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        
        # Create cuda stream for asynchronous copies and inference
        self.stream = cuda.Stream()

        # --- TENSORRT 10.x COMPATIBLE TENSOR PARSING ---
        # Instead of 'bindings', we iterate through names using engine.num_io_tensors
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(tensor_name)
  
        # Raise error if input length > 1 (this wrapper expects a single input tensor)
        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input, found {len(self.input_names)}.")
       
        self.input_name = self.input_names[0]      # Main image input tensor name

    def infer(self, input_array: np.ndarray):   
        if input_array.ndim != 4: 
            raise ValueError(f"Expected BCHW input, got shape {input_array.shape}.")

        # Tell TensorRT the actual shape of the input if the engine uses dynamic input shapes
        # TensorRT 10 uses get_tensor_shape instead of get_binding_shape
        if -1 in tuple(self.engine.get_tensor_shape(self.input_name)):
            self.context.set_input_shape(self.input_name, input_array.shape)
        
        # In TRT 10.x, we must explicitly set tensor addresses on the context
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        input_array = np.ascontiguousarray(input_array.astype(input_dtype, copy=False))

        # Allocate GPU memory for the input and copy host to device
        d_input = cuda.mem_alloc(input_array.nbytes)
        cuda.memcpy_htod_async(d_input, input_array, self.stream)       
        self.context.set_tensor_address(self.input_name, int(d_input))

        # Lists to keep track of output buffers on CPU and GPU
        cpu_outputs = []
        gpu_outputs = []
        output_shapes = []

        # Allocate memory for every output tensor using TRT 10 methods
        for name in self.output_names:
            # Obtain the output shape from the execution context
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Output shape still dynamic for tensor {name}: {shape}.")
            
            # Determine output dtype and total number of elements
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))

            # Create CPU memory buffer and GPU destination memory
            cpu_mem = cuda.pagelocked_empty(size, dtype)
            gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
            
            self.context.set_tensor_address(name, int(gpu_mem))
            cpu_outputs.append(cpu_mem)
            gpu_outputs.append(gpu_mem)
            output_shapes.append(shape)
        
        # Run inference using the modern execute_async_v3 API (v2 is deprecated)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy each output back from gpu to cpu
        for cpu_mem, gpu_mem in zip(cpu_outputs, gpu_outputs):
            cuda.memcpy_dtoh_async(cpu_mem, gpu_mem, self.stream)
        
        # Wait until all queued CUDA work has finished
        self.stream.synchronize()

        # Reshape flat outputs into their original tensor shape
        outputs = [cpu.reshape(shape) for cpu, shape in zip(cpu_outputs, output_shapes)]
        return outputs
    
class RTMPose:
    def __init__(self, engine, frame_dir):
        self.engine = TRTEngine(engine)
        self.frame_dir = Path(frame_dir)        # create the directory containing the FrameIn / FrameOut folders
        self.input_w, self.input_h = (256, 256)
        self.vis_dir = self.frame_dir / "FrameOut"
        self.conf = 0.12

    def read_image(self, image_array):
        if isinstance(image_array, (str, Path)):
            img = cv2.imread(str(image_array), cv2.IMREAD_COLOR)        # loads the images in BGR
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image_array}.")
            return img

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
        """
        RTMPose exports take different output forms depending on model export
        Supported Forms:
        1) One Output: (B, K, 2) or (B, K, 3) -> Direct Coordinates
        2) One Output: (B, K, H, W) -> Heatmaps
        3) Two Outputs: (B, K, L) -> SimCC-style x,y distributions
        """

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
            
    def get_keypoints(self):
        results = []

        img = cv2.imread(str(self.frame_dir / "FrameIn" / "img.jpg"), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        half = w // 2

        left  = img[:, :half]
        right = img[:, half:]  
        image_paths = [left, right]
        n = 0
        for image_path in image_paths:
            image = self.read_image(image_path)

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
            vis = self.draw_hand(image, kp_coords, scores)
            out =  str(n) + ".jpeg"
            cv2.imwrite(str(self.vis_dir / out), vis)
            n += 1
        
        return results

if __name__ == "__main__":
    # Hardcoded paths for this specific machine/project layout.
    ENGINE = "/content/rtmpose_onnx/end2end.engine"
    FRAME_DIR = "/content/Hand-Tracking-2/VideoTracking/"

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    print("TensorRT version used at runtime:", trt.__version__)

    # --- SAFE LIFECYCLE WRAPPING ---
    # Nesting inside functions or using direct context blocks stops garbage-collection races
    def run_pipeline():
        # Create the model wrapper.
        # This internally instantiates TRTEngine and hooks up the CUDA bindings.
        pose = RTMPose(engine=ENGINE, frame_dir=FRAME_DIR)

        # Process keypoints
        kps = pose.get_keypoints()
        left_kps = kps[0]
        right_kps = kps[1]

        print("Keypoints parsed successfully from left and right frames.")

        # --- Clean out local object allocations explicitly ---
        # This guarantees Python releases GPU memory links *before* PyCUDA cleans the context
        del pose
        
    # Execute everything within a functional boundary
    run_pipeline()

    print("Cleanup complete. Exiting smoothly without driver memory leaks!")