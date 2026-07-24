import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print(
    "Device name:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
)

# Check what device the Trainer actually picked
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

import torch_geometric

print("torch_geometric version:", torch_geometric.__version__)
print("torch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
