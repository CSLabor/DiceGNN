from abc import abstractmethod
import datetime
import itertools
from dataclasses import dataclass, fields
#from collections.abc import Iterable, Iterator, Sized
from typing import Iterable, Iterator, Sized
from typing import List, Optional, NamedTuple
import torch
from torch_sparse import SparseTensor

import fast_sampler
from .monkeypatch import Adj
import dgl
from dgl.heterograph import DGLBlock

#from torch.profiler import record_function

def Adj__from_fast_sampler(adj) -> Adj:
    rowptr, col, e_id, sparse_sizes = adj
    return Adj(
        SparseTensor(rowptr=rowptr, row=None, col=col, value=None,
                     sparse_sizes=sparse_sizes, is_sorted=True,
                     trust_data=True),
        e_id,
        sparse_sizes[::-1]
    )

def DGLBlock__from_fast_sampler(adj) -> DGLBlock:
    rowptr, col, e_id, sparse_sizes = adj
    #assert rowptr.dtype == torch.int64
    #dgl_adj = dgl.create_block(('csc', (rowptr, col, eid)), num_dst_nodes=size[1], num_src_nodes=size[0])
    # TODO dont know why, without clone the program throw c10 memory error
    dgl_adj = dgl.create_block(('csc', (rowptr.clone(), col.clone(), torch.Tensor())), 
            num_dst_nodes=sparse_sizes[0], num_src_nodes=sparse_sizes[1],
            idtype=torch.int64, device=torch.device('cpu'))
    #dgl_adj.pin_memory_()
    return dgl_adj


class ProtoBatch(NamedTuple):
    n_id: torch.Tensor
    adjs: List[Adj]
    idx_range: slice

    @classmethod
    def from_fast_sampler(cls, proto_sample):  # , idx_range):
        n_id, adjs, (start, stop) = proto_sample
        adjs = [Adj__from_fast_sampler(adj) for adj in adjs]
        return cls(n_id=n_id, adjs=adjs, idx_range=slice(start, stop))

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start

class PreparedRawBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    adjs: List#: List[DGLBlock]

    @classmethod
    def from_fast_sampler(cls, prepared_sample):
        x, y, adjs, (start, stop) = prepared_sample
        return cls(
            x=x,
            y=y.squeeze(),
            adjs=adjs
        )
        # [[torch.tensor(item, dtype=torch.int64, device=x.device) 
        # if isinstance(item, tuple) else item for item in adj] for adj in adjs]#[DGLBlock__from_fast_sampler(adj) for adj in adjs]

    def record_stream(self, stream):
        self.x.record_stream(stream)
        self.y.record_stream(stream)
        for adj in self.adjs:
            for item in adj:
                if not isinstance(item, tuple):
                    item.record_stream(stream)

    def to(self, device, non_blocking=False):
        return PreparedRawBatch(
            x=self.x.to(
                device=device,
                non_blocking=non_blocking),
            y=self.y.to(
                device=device,
                non_blocking=non_blocking),
            adjs = [[item.to(device=device, non_blocking=non_blocking) if not isinstance(item, tuple)
                else item for item in adj] for adj in self.adjs],
        )


class PreparedBatch(NamedTuple):
    x: torch.Tensor
    y: Optional[torch.Tensor]
    adjs: List[Adj]
    idx_range: slice

    @classmethod
    def from_proto_batch(cls, x: torch.Tensor,
                         y: Optional[torch.Tensor],
                         proto_batch: ProtoBatch):
        return cls(
            x=x[proto_batch.n_id],
            y=y[proto_batch.n_id[:proto_batch.batch_size]
                ] if y is not None else None,
            adjs=proto_batch.adjs,
            idx_range=proto_batch.idx_range
        )

    @classmethod
    def from_fast_sampler(cls, prepared_sample):
        x, y, adjs, (start, stop) = prepared_sample
        return cls(
            x=x,
            y=y.squeeze() if y is not None else None,
            adjs=[Adj__from_fast_sampler(adj) for adj in adjs],
            idx_range=slice(start, stop)
        )

    def record_stream(self, stream):
        if self.x is not None:
            self.x.record_stream(stream)
        if self.y is not None:
            self.y.record_stream(stream)
        for adj in self.adjs:
            adj.record_stream(stream)

    def to(self, device, non_blocking=False):
        return PreparedBatch(
            x=self.x.to(
                device=device,
                non_blocking=non_blocking) if self.x is not None else None,
            y=self.y.to(
                device=device,
                non_blocking=non_blocking) if self.y is not None else None,
            adjs=[adj.to(device=device, non_blocking=non_blocking)
                  for adj in self.adjs],
            idx_range=self.idx_range
        )

    @property
    def num_total_nodes(self):
        return self.x.size(0)

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


@dataclass
class FastSamplerConfig:
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    idx: torch.Tensor
    batch_size: int
    sizes: List[int]
    skip_nonfull_batch: bool
    pin_memory: bool

    def to_fast_sampler(self) -> fast_sampler.Config:
        c = fast_sampler.Config()
        for field in fields(self):
            setattr(c, field.name, getattr(self, field.name))

        return c

    def get_num_batches(self) -> int:
        num_batches, r = divmod(self.idx.numel(), self.batch_size)
        if not self.skip_nonfull_batch and r > 0:
            num_batches += 1
        return num_batches


class FastSamplerStats(NamedTuple):
    total_blocked_dur: datetime.timedelta
    total_blocked_occasions: int

    @classmethod
    def from_session(cls, session: fast_sampler.Session):
        return cls(total_blocked_dur=session.total_blocked_dur,
                   total_blocked_occasions=session.total_blocked_occasions)


#class FastSamplerIter(Iterator[PreparedBatch]):
class FastSamplerIter(Iterator[PreparedRawBatch]):
    session: fast_sampler.Session

    def __init__(self, num_threads: int, max_items_in_queue:
                 int, cfg: FastSamplerConfig):
        ncfg = cfg.to_fast_sampler()
        self.session = fast_sampler.Session(
            num_threads, max_items_in_queue, ncfg)
        #assert self.session.num_total_batches == cfg.get_num_batches()
        self.pos = 0
        self.length = cfg.get_num_batches()

    def check_release(self):
        if self.pos == self.length:
            del self.session

    def try_one(self):
        #if self.pos == self.length:
        if self.session.num_consumed_batches == self.session.num_total_batches:
            del self.session
            raise StopIteration
        sample = self.session.try_get_batch()
        if sample is None:
            return None
        self.pos += 1
        #self.check_release()
        #return PreparedBatch.from_fast_sampler(sample)
        return PreparedRawBatch.from_fast_sampler(sample)

    def __next__(self):
        sample = self.session.blocking_get_batch()
        if sample is None:
            del self.session
            raise StopIteration
        self.pos += 1
        #self.check_release()
        #return PreparedBatch.from_fast_sampler(sample)
        return PreparedRawBatch.from_fast_sampler(sample)

    def get_stats(self) -> FastSamplerStats:
        return FastSamplerStats.from_session(self.session)


class ABCNeighborSampler(Iterable[PreparedBatch], Sized):
    @property
    @abstractmethod
    def idx(self) -> torch.Tensor:
        ...

    @idx.setter
    @abstractmethod
    def idx(self, idx: torch.Tensor) -> None:
        ...


@dataclass
class FastSampler(ABCNeighborSampler):
    num_threads: int
    max_items_in_queue: int
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    def __iter__(self):
        return FastSamplerIter(self.num_threads, self.max_items_in_queue,
                               self.cfg)

    def __len__(self):
        return self.cfg.get_num_batches()


@dataclass
class FastPreSampler(ABCNeighborSampler):
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    def __iter__(self) -> Iterator[PreparedBatch]:
        cfg = self.cfg
        p = fast_sampler.full_sample(cfg.x, cfg.y, cfg.rowptr, cfg.col,
                                     cfg.idx, cfg.batch_size, cfg.sizes,
                                     cfg.skip_nonfull_batch, cfg.pin_memory)
        return (PreparedBatch.from_fast_sampler(sample)
                for sample in itertools.chain(*p))

    def __len__(self):
        return self.cfg.get_num_batches()
