import cv2
import warnings 
import numpy as np
import pycuda.autoinit      # Initializes cuda when imported
import tensorrt as trt
from pathlib import Path
import pycuda.driver as cuda

# Hide the non-critical warnings to keep console clean
warnings.filterwarnings("ignore")

# Define the Hand Skeleton connections. Each tuple draws a line between keypoints A and B
# The keypoints indices match the model's output ordering.

HAND_SKELETON = [ 
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

class TRTEngine:
    def __init__(self, engine):
        self.engine_path = engine

        self.logger = trt.Logger(trt.Logger.ERROR)      # Create TensorRT logger that only prints errors
        trt.init_libnvinfer_pluggins(self.logger, "")   # Load any TensorRT plugins that are required by the engine

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
        self.output_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
  
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
            dtype = trt.ntype(self.engine.get_binding_dtype(idx))
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
    def __init__(self, engine, frame_dir, vis_dir, conf, input_size = (256, 256)):
        self.engine = TRTEngine(engine)
        self.frame_dir = Path(frame_dir)        # create the directory containing the FrameIn / FrameOut folders
        self.input_w, self.input_h = input_size
        self.vis_dir = vis_dir
        self.conf = conf

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
        np.transpose(img, (2, 0, 1))
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
            cv2.line(vis, pt1, pt2, (41, 212, 166), 2)      # Color the lines of the keypoint skeleton

        # Draw each keypoint as filled circle with border
        for idx, pt in coords.items():
            if scores[idx] < self.conf:
                continue

            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(vis, center, 4, (0, 255, 0), -1)
            cv2.circle(vis, center, 4, (0, 0, 0), 1)    
        
        return vis
            
    def get_keypoints(self):
        None
    
    def get_coords(self):
        results = self.get_keypoints()
        img_coords = []

        for coords in results:
            img_coords.append(coords['keypoints'])
        
        return img_coords
    

# Complete get_keypoints and decode_outputs functions.