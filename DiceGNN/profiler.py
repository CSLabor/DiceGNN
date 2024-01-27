from .utils import get_logger
mlog = get_logger()

import os
import dgl
import math
import time
import torch
import random
import numpy as np

def measure_gpu_batching_time(num_trials, gpu_loader):
    """
    return average gpu batching time in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for GPU Batching")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with other ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            start_events[i].record()
            ret = next(gpu_iter)
            end_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        #mlog(elapsed_times[:5])
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs[1:]) if len(avgs) > 1 else avgs[0]

def measure_cpu_batching_time(num_trials, cpu_loader):
    """
    return average cpu batching time in ms
    cpu warmup is slow, so the first run is discarded from stats
    """
    from time import perf_counter
    avgs = []
    mlog(f"\n=======")
    for r in range(1+num_trials):
        if r:
            mlog(f"Profile RUN {r} for CPU Batching ")
        else:
            mlog(f"Warmup for CPU Batching")
        tmp_cnt = 0
        cpu_iter = iter(cpu_loader)
        durs = []
        st = perf_counter()
        while True:
            try:
                ret = cpu_iter.try_one()
                if ret is None:
                    continue
                ed = perf_counter()
                durs.append(1000*(ed-st))
                st = perf_counter()
                tmp_cnt += 1
                if tmp_cnt > 100 and r == 0:
                    # first warmup we run for 100 batches only
                    del cpu_iter
                    break
            except StopIteration:
                break
        if r:
            #mlog(durs[:5])
            mlog(f"{np.mean(durs):.2f} ± {np.std(durs):.2f} ms/batch")
            avgs.append(np.mean(durs))
        else:
            mlog(f"Warmup finish for CPU Batching")
    return np.mean(avgs)
 
def measure_dma_transfering_time(cpu_loader):
    """
    return DMA transferring time in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    mlog("measuring DMA transferring time")
    num_batches = 50
    saved_batches = []
    while len(saved_batches) < num_batches:
        cpu_iter = iter(cpu_loader)
        for ret in cpu_iter:
            saved_batches.append(ret)
            if len(saved_batches) == num_batches:
                break
    # DMA running time is very stable, so we only give one run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    # first saturate GPU and PCIe
    for batch in saved_batches[-10:]:
        batch.to(device, non_blocking=True)

    # then time
    for i, batch in enumerate(saved_batches):
        start_events[i].record()
        batch.to(device, non_blocking=True)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    #mlog(elapsed_times[:5])
    mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
    return np.mean(elapsed_times)

def gpu_batch_core(gpu_iter):
    partial = next(gpu_iter)
    if len(partial) != 3:
        partial = gpu_iter.fetch_partial_batch(partial)
    return partial

def measure_gpu_batching_model_time(num_trials, gpu_loader, model, loss_fn, optimizer):
    """
    return:
    * GPU batching time
        * partial sampling time
        * fetching cache time
    * Model training time 
    in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for GPU batching and model")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        batching_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        if gpu_iter.with_cache:
            cache_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with some ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            start_events[i].record()
            batch = next(gpu_iter)
            batching_events[i].record()
            if gpu_iter.with_cache:
                batch = gpu_iter.fetch_partial_batch(batch)
                cache_events[i].record()
            batch_x, batch_y, adjs = batch
            batch_pred = model(adjs, batch_x)
            loss = loss_fn(batch_pred, batch_y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_events[i].record()
        torch.cuda.synchronize()

        cur_ret = []
        batching_times = [s.elapsed_time(e) for s, e in zip(start_events, batching_events)][1:]
        cur_ret.append(np.mean(batching_times))
        if gpu_iter.with_cache:
            cache_times = [s.elapsed_time(e) for s, e in zip(batching_events, cache_events)][1:]
            model_times = [s.elapsed_time(e) for s, e in zip(cache_events, end_events)][1:]
            cur_ret.append(np.mean(cache_times))
        else:
            model_times = [s.elapsed_time(e) for s, e in zip(batching_events, end_events)][1:]
        cur_ret.append(np.mean(model_times))

        mlog(f"gpu_batching: {np.mean(batching_times):.2f} ± {np.std(batching_times):.2f} ms/batch")
        if gpu_iter.with_cache:
            mlog(f"cache fetching: {np.mean(cache_times):.2f} ± {np.std(batching_times):.2f} ms/batch")
        mlog(f"model training: {np.mean(model_times):.2f} ± {np.std(model_times):.2f} ms/batch")

        avgs.append(cur_ret)
    return list(np.mean(avgs, axis=0))


def measure_model_training_time(num_trials, gpu_loader, model, loss_fn, optimizer):
    """
    return model training time in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for Model")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with some ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            batch_x, batch_y, adjs = gpu_batch_core(gpu_iter)
            start_events[i].record()
            batch_pred = model(adjs, batch_x)
            loss = loss_fn(batch_pred, batch_y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        #mlog(elapsed_times[:5])
        elapsed_times = elapsed_times[1:]
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs)

if __name__ == '__main__':
    pass
