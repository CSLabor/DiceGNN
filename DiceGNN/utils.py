import dgl
import torch
import random
import logging
import numpy as np

def get_logger(file_path=None):
    if file_path:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.INFO,
            filename=file_path,
            filemode='w'
        )
        print("Logs are being recorded at: {}".format(file_path))
    else:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.CRITICAL
        )
    log = logging.getLogger(__name__).critical
    return log

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    dgl.random.seed(seed)

def truncate(int_, min_, max_):
    if int_ > max_:
        return max_
    if int_ < min_:
        return min_
    return int_

