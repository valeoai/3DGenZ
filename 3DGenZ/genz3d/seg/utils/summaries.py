import os

import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid



class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(os.path.join(self.directory))
        return writer