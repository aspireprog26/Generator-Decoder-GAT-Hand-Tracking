import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience, min_delta, decoder_path, generator_path):
        self.patience = patience
        self.min_delta = min_delta
        self.decoder_path = decoder_path
        self.generator_path = generator_path

        self.best_loss = np.inf
        self.counter = 0
        self.stopping = False

    def __call__(self, val_loss, decoder_model, generator_model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(decoder_model.state_dict(), self.decoder_path)
            torch.save(generator_model.state_dict(), self.generator_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopping = True
