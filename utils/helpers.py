import os
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dirs(dir_paths):
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)