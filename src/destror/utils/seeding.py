import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for random, numpy, and torch for deterministic runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # CUDA ops may still be non-deterministic; log this fact
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
