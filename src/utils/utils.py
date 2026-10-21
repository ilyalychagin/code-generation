import numpy as np
import torch
import random
import os

def set_seed(seed: int): # TODO: возможно предстоит сделать более расширенную установку seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)