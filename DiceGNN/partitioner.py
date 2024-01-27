import math
from .utils import *

def initial_guess(n_total, t_cpu, t_dma, t_model, t_gpu, t_cache):
    # initial guess, a monotone min max problem of three lines
    y1 = lambda x : x * t_cpu                                                   # CPU timeline function
    y2 = lambda x : x * t_dma + (n_total - x) * t_gpu                           # PCIe timeline function
    y3 = lambda x : n_total * t_model + (n_total - x) * (t_gpu + t_cache)       # GPU timeline function

    x1 = n_total * t_gpu / (t_cpu + t_gpu - t_dma)                              # cpu time = pcie time
    x2 = n_total * (t_model + t_gpu + t_cache) / (t_gpu + t_cache + t_cpu)      # cpu time = gpu time
    x3 = n_total * (t_model + t_cache) / (t_dma - t_cache)                      # gpu time = pcie time

    record_min_time = n_total * (t_cpu + t_model + t_dma + t_gpu + t_cache)
    record_n_cpu = -1
    for x in [x1, x2, x3]:
        low, high = truncate(math.floor(x), 0, n_total), truncate(math.ceil(x), 0, n_total)
        for ncpu in [low, high]:
            cur_max = max(y1(ncpu), y2(ncpu), y3(ncpu))
            if cur_max < record_min_time:
                record_min_time = cur_max
                record_n_cpu = ncpu

    return record_n_cpu, n_total - record_n_cpu


def tune_with_feedback(feedback, oldplan):
    # finetune oldplan with feedback
    if feedback == 1:
        # too much cpu workload 
        return oldplan[0] - 1, oldplan[1] + 1
    # too much gpu workload
    return oldplan[0] + 1, oldplan[1] - 1
