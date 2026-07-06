import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience, min_delta, path):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        
        self.best_loss = np.inf
        self.counter = 0
        self.stopping = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1 
            if self.counter >= self.patience:
                self.stopping = True