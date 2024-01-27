from DiceGNN.utils import get_logger
mlog = get_logger()

from third_party.salient.fast_trainer.samplers import *
from third_party.salient.fast_trainer.transferers import *

import dgl
import math
import torch
from dgl.utils.pin_memory import gather_pinned_tensor_rows

def prepare_salient(x, y, row, col, train_idx, train_batch_size, num_workers, train_fanouts):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    cfg = FastSamplerConfig(
        x=x, y=y,
        rowptr=row, col=col,
        idx=train_idx,
        batch_size=train_batch_size, sizes=train_fanouts,
        skip_nonfull_batch=False, pin_memory=True
    ) 
    train_max_num_batches = cfg.get_num_batches()
    cpu_loader = FastSampler(num_workers, train_max_num_batches, cfg)
    mlog('SALIENT CPU batcher prepared')
    return cpu_loader

def prepare_dgl_gpu(graph, all_data, sampler, train_idx, train_batch_size):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    train_idx = train_idx.cuda()
    gpu_loader = DGL_GPU_iter(graph, sampler, all_data, train_batch_size, train_idx)
    mlog('DGL GPU batcher prepared')
    return gpu_loader

def prepare_dgl_cache(graph, all_data, sampler, train_idx, train_batch_size, all_cache, gpu_flag, gpu_map):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    train_idx = train_idx.cuda()
    gpu_loader = DGL_GPU_Cache_iter(graph, sampler, all_data, train_batch_size, train_idx, all_cache, gpu_flag, gpu_map)
    mlog('DGL Cache batcher prepared')
    return gpu_loader

class Blank_iter(Iterator):
    def __init__(self):
        self.length = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    def __len__(self):
        return self.length
       

class DGL_GPU_iter(Iterator):
    def __init__(self, graph, sampler, all_data, bs, train_idx):
        self.graph = graph
        self.sampler = sampler
        self.all_data = all_data
        self.bs = bs
        self._idx = train_idx.cuda()
        self.length = math.ceil(self._idx.shape[0] / self.bs)
        self.with_cache = False

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, new_idx: torch.Tensor):
        self._idx = new_idx.cuda()
        self.length = math.ceil(self._idx.shape[0] / self.bs)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        st = self.pos * self.bs
        ed = min(self._idx.shape[0], st + self.bs)
        self.pos += 1
        cur_seeds = self._idx[st:ed]
        input_nodes, output_nodes, blocks = self.sampler.sample(self.graph, cur_seeds)
        cur_x = gather_pinned_tensor_rows(self.all_data[0], input_nodes)
        cur_y = gather_pinned_tensor_rows(self.all_data[1], output_nodes)
        return cur_x, cur_y, blocks

class DGL_GPU_Cache_iter(Iterator):
    def __init__(self, graph, sampler, all_data, bs, train_idx, all_cache, gpu_flag, gpu_map):
        self.graph = graph
        self.sampler = sampler
        self.all_data = all_data
        self.bs = bs
        self._idx = train_idx.cuda()
        self.all_cache = all_cache
        self.gpu_flag = gpu_flag
        self.gpu_map = gpu_map
        self.length = math.ceil(self._idx.shape[0] / self.bs)
        self.nfeat_buf = None
        self.label_buf = None
        self.with_cache = True

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, new_idx: torch.Tensor):
        self._idx = new_idx.cuda()
        self.length = math.ceil(self._idx.shape[0] / self.bs)

    def load_from_cache(self, cpu_partial, gpu_orig, idx, out_buf):
        gpu_mask = self.gpu_flag[idx]
        gpu_nids = idx[gpu_mask]
        gpu_local_nids = self.gpu_map[gpu_nids].long()
        cur_res = out_buf[:idx.shape[0]]
        cur_res[gpu_mask] = gpu_orig[gpu_local_nids]
        cur_res[~gpu_mask] = cpu_partial
        return cur_res

    def fetch_partial_batch(self, partial_batch):
        input_nodes, cur_bs, cpu_x, cpu_y, blocks = partial_batch
        if self.nfeat_buf is None:
            # create
            self.nfeat_buf = torch.zeros((int(1.5*input_nodes.shape[0]), cpu_x.shape[1]), dtype=cpu_x.dtype, device=torch.device('cuda:0'))
            self.label_buf = torch.zeros((self.bs, 1), dtype=cpu_y.dtype, device=torch.device('cuda:0'))

        if self.nfeat_buf.shape[0] < input_nodes.shape[0]:
            # resize
            mlog('resizing buffer')
            del self.nfeat_buf, self.label_buf
            self.nfeat_buf = torch.zeros((int(1.2*input_nodes.shape[0]), cpu_x.shape[1]), dtype=cpu_x.dtype, device=torch.device('cuda:0'))
            self.label_buf = torch.zeros((self.bs, 1), dtype=cpu_y.dtype, device=torch.device('cuda:0'))

        ret_x = self.load_from_cache(cpu_x, self.all_cache[0], input_nodes, self.nfeat_buf)
        ret_y = self.load_from_cache(cpu_y, self.all_cache[1], input_nodes[:cur_bs], self.label_buf)
        return ret_x, ret_y, blocks

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        st = self.pos * self.bs
        ed = min(self._idx.shape[0], st + self.bs)
        self.pos += 1
        cur_seeds = self._idx[st:ed]
        input_nodes, output_nodes, blocks = self.sampler.sample(self.graph, cur_seeds)
        # only fetch partial data on CPU, leave the GPU part for later process
        cpu_mask = ~self.gpu_flag[input_nodes]
        cpu_x = gather_pinned_tensor_rows(self.all_data[0], input_nodes[cpu_mask])
        cpu_y = gather_pinned_tensor_rows(self.all_data[1], output_nodes[cpu_mask[:(ed-st)]])

        # partial batch format: input_nodes, batch_size, cpu_x, cpu_y, blocks
        return input_nodes, ed-st, cpu_x, cpu_y, blocks

