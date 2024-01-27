from DiceGNN.utils import get_logger
mlog = get_logger()

import dgl
import math
import torch
from dgl.utils.pin_memory import gather_pinned_tensor_rows

def construct_cache(graph, train_idx, sampler, bs, nfeat, label, budget):
    """
    first get frequency with sampler, then calculate budget and the cache
    """
    train_idx = train_idx.cuda()
    nfeat_counts = torch.zeros(nfeat.shape[0], dtype=torch.int32, device=torch.device('cuda:0'))
    for _ in range(2):
        it = tmp_iter(train_idx, bs)
        for seed in it:
            input_nodes, _, _ = sampler.sample(graph, seed)
            nfeat_counts[input_nodes] += 1
    mlog('get counts')
    prio_idx_order = nfeat_counts.argsort(descending=True) # on GPU
    

    # lines to cache
    single_line_bytes = nfeat.element_size() * nfeat.shape[1] + label.element_size()
    n_lines = int(budget * 1024**3 / single_line_bytes)
    cache_nids = prio_idx_order[:n_lines]
    mlog(f'cache numbers: {n_lines}')

    # prepare flag
    gpu_flag = torch.zeros(nfeat.shape[0], dtype=torch.bool, device=torch.device('cuda:0'))
    gpu_flag[cache_nids] = True

    # prepare cache
    #all_cache = [gather_pinned_tensor_rows(nl, cache_nids) for nl in [nfeat, label]]
    all_cache = [nl[cache_nids.cpu()].cuda() for nl in [nfeat, label]]

    # prepare map in GPU
    gpu_map = torch.zeros(nfeat.shape[0], dtype=torch.int32, device=torch.device('cuda:0')).fill_(-1)
    gpu_map[cache_nids] = torch.arange(cache_nids.shape[0], dtype=torch.int32, device=torch.device('cuda:0'))

    mlog('get map/flag')
    del nfeat_counts, input_nodes, prio_idx_order, cache_nids
    return all_cache, gpu_flag, gpu_map


def tmp_iter(train_idx, bs):
    pm = torch.randperm(train_idx.shape[0]).to(train_idx.device)
    local_train_idx = train_idx[pm]
    length = math.ceil(train_idx.shape[0] / bs)
    for i in range(length):
        st = i * bs
        ed = min((i+1) * bs, train_idx.shape[0])
        yield local_train_idx[st:ed]
