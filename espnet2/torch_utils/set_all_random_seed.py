import random

import numpy as np
import oneflow as torch


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
