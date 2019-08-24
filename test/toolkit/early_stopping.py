import os
import torch

class EarlyStopping():
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 100.0
        self.early_stop = False
        self.best_model_path = None
        weight_path = os.path.join("exp", "Ori_model", "weights")
        if not os.path.exists(weight_path): os.makedirs(weight_path)

    def __call__(self, val_loss, model):
        score = val_loss
        
        if score <= self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter =self.counter + 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print("Model improved. Checkpoint saved.")
            
        torch.save(model.state_dict(), os.path.join("exp", "Ori_model", "weights", "weights.pt"))
        self.best_model_path = os.path.join("exp", "Ori_model", "weights", "weights.pt")

