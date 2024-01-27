#from DiceGNN.utils import get_logger
#mlog = get_logger()
def mlog(arg):
    pass

import dgl
import torch
from collections import deque

class DiceScheduledTrainer:
    """
    given data (CPU&GPU iterators), model, constrains (gpu&dma buffer size)
    reschedule the whole training procedure
    """
    def __init__(self, device, cpu_it, gpu_it, model, opt, loss_fn, buffer_size, dma_size):
        """
        arguments:
            device: the gpu
            cpu_it: FastSamplerIter, should be derived from FastSampler or manually controlled (wrt train_idx)
            gpu_it: DGL-UVA sampler, should be mannually controlled wrt train_idx
            model:  the GNN model
            opt:    the optimizer
            loss_fn:   the loss function
            buffer_size: maximum number of mini-batches that resides in GPU
            dma_size: number of mini-batches to be transferred for each DMA phase

        expected behavior:
            during training, try best to overlap the DMA phase with the Model phase
            specifically, we first try to fill the Buffer with gpu_it, and wait for the time when CPU prepare
            enough (#dma_size) mini-batches, then, we concurrently execute DMA transfer and the Model training

        about gpu buffer size:
            when gpu buffer is full, the gpu can still hold one extra mini-batch

        """
        self.device = device
        self.cpu_it = cpu_it
        self.gpu_it = gpu_it
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.buffer_size = buffer_size
        self.dma_size = dma_size

        self.gpu_buffer = deque()
        self.dma_buffer = deque()
        self.copy_stream = torch.cuda.Stream(self.device)
        self.next_batch = None

        self.flag_cpu_end = self.cpu_it.length == 0
        self.flag_gpu_end = self.gpu_it.length == 0

    def train_one_batch(self, batch, event=None):
        cur_stream = torch.cuda.current_stream(self.device)
        if event is not None:
            mlog('train one cpu buffer batch')
            # https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html
            cur_stream.wait_event(event)
            batch.record_stream(cur_stream)
            batch = construct_dgl_block(batch)
        else:
            mlog('train one gpu/cpu_it batch')

        if len(batch) == 3:
            batch_x, batch_y, adjs = batch
        else:
            batch_x, batch_y, adjs = self.gpu_it.fetch_partial_batch(batch)

        batch_pred = self.model(adjs, batch_x)
        loss = self.loss_fn(batch_pred, batch_y.reshape(-1))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        del batch_x, batch_y, adjs, batch_pred

    def preload_cpu(self, source):
        assert source in ['cpu_it', 'dma_buffer']
        self.next_batch = None
        if source == 'cpu_it':
            if self.flag_cpu_end:
                return
            batch = next(self.cpu_it)
            self.flag_cpu_end = self.cpu_it.pos == self.cpu_it.length
            mlog('Preload: cpu batch from iterator')
        else: # source == 'dma_buffer'
            if len(self.dma_buffer) == 0:
                return
            mlog('Preload: cpu batch from buffer')
            batch = self.dma_buffer.popleft()

        with torch.cuda.stream(self.copy_stream):
            self.next_batch = batch.to(self.device, non_blocking=True)

    def overlap_dma_gpu_buffer(self):
        cur_stream = torch.cuda.current_stream(self.device)
        """
        **we want to flush two buffers**, but in the same time
        we try to overlap Model phase and DMA phase
        we assume DMA time > Model time

        for loop: 
            1. pop out one batch from buffer
            2. initiate one DMA mini-batch transfer (if still have), append to buffer
            3. initiate model training on the mini-batch just popped from the buffer
            4. delete the mini-batch

        the memory consumption is steady or gradually smaller because DMA transfer is
        slower than model, and dma_buffer is smaller than gpu_buffer
        """
        mlog('Overlap')
        if len(self.gpu_buffer) + len(self.dma_buffer) == 0:
            return

        if len(self.dma_buffer) == 0:
            # directly empty gpu_buffer
            while len(self.gpu_buffer):
                event, batch = self.gpu_buffer.popleft()
                self.train_one_batch(batch, event)
            return

        if len(self.gpu_buffer) == 0:
            # cpu preload buffer and model
            if self.next_batch is None:
                self.preload_cpu(source='dma_buffer')
            while not (self.next_batch is None and len(self.dma_buffer) == 0):
                cur_stream.wait_stream(self.copy_stream)
                ret = self.next_batch
                self.preload_cpu(source='dma_buffer')
                ret.record_stream(cur_stream)
                ret = construct_dgl_block(ret)
                self.train_one_batch(ret)
            return

        # both buffers are not empty
        temp_flag = True
        while len(self.gpu_buffer):
            # 1. pop
            cur_event, cur_batch = self.gpu_buffer.popleft()

            # 2. dma batches append to gpu buffer tail
            if len(self.dma_buffer):
                cpu_batch = self.dma_buffer.popleft()
                if temp_flag:
                    self.copy_stream.wait_stream(cur_stream) # avoid DMA conflict with UVA on PCIE
                    temp_flag = False
                with torch.cuda.stream(self.copy_stream):
                    future_batch = cpu_batch.to(self.device, non_blocking=True)
                future_event = self.copy_stream.record_event()
                self.gpu_buffer.append((future_event, future_batch))

            # 3. model batch from gpu buffer head
            self.train_one_batch(cur_batch, cur_event)

            # 4. delete
            del cur_batch


    def train_one_epoch(self):
        cur_stream = torch.cuda.current_stream(self.device)
        while not (self.flag_gpu_end and self.flag_cpu_end and len(self.dma_buffer) == 0 and len(self.gpu_buffer) == 0):
            if self.flag_gpu_end and self.flag_cpu_end: # flush two buffers
                mlog('Flush')
                self.overlap_dma_gpu_buffer()
                break
            elif self.flag_cpu_end: # finish gpu_it and flush buffer
                mlog('GPUOnly')
                while not self.flag_gpu_end:
                    batch = next(self.gpu_it)
                    self.flag_gpu_end = self.gpu_it.pos == self.gpu_it.length
                    self.train_one_batch(batch)
                self.overlap_dma_gpu_buffer()
                break
            elif self.flag_gpu_end: # finish cpu_it and flush buffer
                mlog('CPUOnly')
                if self.next_batch is None:
                    self.preload_cpu(source='cpu_it')
                while not (self.next_batch is None and self.flag_cpu_end):
                    cur_stream.wait_stream(self.copy_stream)
                    ret = self.next_batch
                    self.preload_cpu(source='cpu_it')
                    ret.record_stream(cur_stream)
                    ret = construct_dgl_block(ret)
                    self.train_one_batch(ret)
                self.overlap_dma_gpu_buffer()
                break
            else: # hybrid scheduling
                # fill gpu buffer
                while len(self.gpu_buffer) < self.buffer_size and not self.flag_gpu_end:
                    mlog('Hybrid: uva fill buffer')
                    self.gpu_buffer.append((None, next(self.gpu_it)))
                    self.flag_gpu_end = self.gpu_it.pos == self.gpu_it.length

                # fill dma buffer
                while len(self.dma_buffer) < self.dma_size and not self.flag_cpu_end:
                    batch = self.cpu_it.try_one()
                    if batch is None:
                        break # no blocking wait for CPU batches
                    mlog('Hybrid: cpu fill buffer')
                    self.dma_buffer.append(batch)
                    self.flag_cpu_end = self.cpu_it.pos == self.cpu_it.length

                if len(self.dma_buffer) == self.dma_size or self.flag_cpu_end:
                    self.overlap_dma_gpu_buffer()
                    continue

                # perform one model, and wait for dma_buffer ready
                mlog('Hybrid: model + wait')
                event, batch = self.gpu_buffer.popleft()
                self.train_one_batch(batch, event)

from typing import Iterator
class AdhocPrefetcher(Iterator):
    """
    #FIXME: have not checked compatability
    """
    def __init__(self, device, cpu_it, gpu_it):
        self.device = device
        self.cpu_it = cpu_it
        self.gpu_it = gpu_it
        self.copy_stream = torch.cuda.Stream(self.device)

        self.flag_cpu_end = False
        self.flag_gpu_end = self.gpu_it.length == 0

        self.next_batch = None
        mlog(f"gpu length: {self.gpu_it.length}, cpu length: {self.cpu_it.length}")
        self.preload()

    def preload(self, cpu_blocking=False):
        self.next_batch = None
        if self.flag_cpu_end:
            return
        try:
            if cpu_blocking:
                batch = next(self.cpu_it)
            else:
                batch = self.cpu_it.try_one()
            if batch is not None:
                with torch.cuda.stream(self.copy_stream):
                    self.next_batch = batch.to(self.device, non_blocking=True)
        except StopIteration:
            self.flag_cpu_end = True
        except Exception as e:
            print(e)
            raise e

    def __next__(self):
        if self.flag_gpu_end and self.flag_cpu_end:
            #torch.cuda.synchronize()
            raise StopIteration

        cur_stream = torch.cuda.current_stream(self.device)

        # CPU batch ready, use it for training
        if not self.flag_gpu_end and self.next_batch and self.copy_stream.query():
            ret = self.next_batch
            mlog(f'1cpu batch, {self.cpu_it.generated_batch}')
            ret = convert_salient_to_dgl(ret)
            record_batch(ret, cur_stream)
            self.preload()
            return ret

        # either next_batch is None or transfer not finished, we first try GPU batching
        if not self.flag_gpu_end:
            batch = next(self.gpu_it)
            mlog(f'1gpu batch, {self.gpu_it.idx}')
            self.flag_gpu_end = self.gpu_it.idx == self.gpu_it.length
            if self.next_batch is None:
                self.preload()
            return batch

        # if GPU end, must be pure CPU batching phase
        if not self.flag_cpu_end:
            if self.next_batch is None:
                self.preload(cpu_blocking=True)
            cur_stream.wait_stream(self.copy_stream)
            ret = self.next_batch
            mlog(f'2cpu batch, {self.cpu_it.generated_batch}')
            ret = convert_salient_to_dgl(ret)
            record_batch(ret, cur_stream)
            self.preload(cpu_blocking=True)
            return ret



from third_party.salient.fast_trainer.samplers import PreparedRawBatch
def construct_dgl_block(raw_batch):
    if not isinstance(raw_batch, PreparedRawBatch):
        return raw_batch
    x, y, adjs = raw_batch
    dgl_blocks = []
    for rowptr, col, e_id, sparse_sizes in adjs:
        dgl_adj = dgl.create_block(('csc', (rowptr, col, torch.empty((0,), dtype=torch.int64, device=torch.device('cuda:0')))), 
            num_dst_nodes=sparse_sizes[0], num_src_nodes=sparse_sizes[1])#)#, idtype=torch.int64, device=torch.device('cuda:0'))
        dgl_blocks.append(dgl_adj)
    return x, y, dgl_blocks

