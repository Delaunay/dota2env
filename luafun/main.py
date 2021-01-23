import torch
import torch.nn as nn


class PPO:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            betas=betas
        )