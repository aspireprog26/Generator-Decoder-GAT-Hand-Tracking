import sys
import torch

sys.path.insert(0, r"/home/mrtcloud-1/Documents/Hand-Tracking-2/Anatomy")
import earlystopper as es
class Trainer:
    def __init__(self, configs, model, train_loader, val_loader):
        self.optimizer = configs["optimizer"]
        self.criterion = configs["criterion"]
        self.model = model
        self.num_epochs = configs["num_epochs"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stop_patience = configs["es_patience"]
        self.min_delta = configs["es_thresh"]
        self.model_save_path = "/home/mrtcloud-1/Documents/Hand-Tracking-2/Anatomy/model.pth"
        self.early_stopper = es.EarlyStopping()

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for input, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for input, target in self.val_loader:
                    output = self.model(input)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)

            print(f"Epoch {epoch} | Train Loss: {train_loss} Val Loss: {val_loss}")
            self.early_stopper(val_loss, self.model)

            if self.early_stopper.stopping:
                print(f"Early Stopping at epoch {epoch} / {self.num_epochs}")
                break