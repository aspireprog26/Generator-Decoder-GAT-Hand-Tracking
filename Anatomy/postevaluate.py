import json
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from model import AnatomyModel
from torch.utils.data import DataLoader

with open("/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Anatomy/configs.json", "r") as f:
    configs = json.load(f)
    
def nmse(target, prediction):
    return np.mean((target - prediction) ** 2) / np.var(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnatomyModel(configs["input_size"], configs["hidden_size"], configs["output_size"], configs["dropout"]).to(device)
weights = torch.load((Path(configs["model_dir"]) / configs["model_name"]), weights_only = True, map_location = device)
model.load_state_dict(weights)
model.eval()

dataset = torch.load(Path(configs["post_data_dir"]) / "Testing" / "dataset.pt")
loader = DataLoader(
    dataset = dataset,
    num_workers = configs["num_workers"],
    batch_size = configs["batch_size"],
    shuffle = True
)

def evalModel():
    test_mse_loss = 0 
    test_nmse_loss = 0
    total_samples = 0

    loss_fn = nn.MSELoss()
    for input, target in loader:
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(input)
            mse_loss = loss_fn(output, target)
            test_mse_loss += mse_loss.item()
            #test_nmse_loss += nmse(output.detach().cpu().numpy(), target.detach().cpu().numpy()) * input.shape[0]
        #total_samples += input.shape[0]

    avg_mse_loss = test_mse_loss / len(loader)
    #avg_nmse_loss = test_nmse_loss / total_samples
    return avg_mse_loss #[avg_mse_loss, avg_nmse_loss]

evaluation = evalModel()
print(f"Average MSE Loss Per Batch: {evaluation: .5f}")