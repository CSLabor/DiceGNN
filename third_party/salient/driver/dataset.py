from typing import Mapping, NamedTuple, Any
from pathlib import Path
import torch
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
import dgl
import os

from fast_sampler import to_row_major

from DiceGNN.utils import get_logger
mlog = get_logger()
import numpy as np

def get_sparse_tensor(edge_index, num_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_nodes, num_nodes)).t()


class FastDataset(NamedTuple):
    name: str
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    split_idx: Mapping[str, torch.Tensor]
    meta_info: Mapping[str, Any]

    @classmethod
    def from_dgl(self, name: str, root=f'/export/data/{os.environ["USER"]}/datasets', fake_dim=256):
        assert name in ['twitter', 'uk', 'mag']
        prep_path = Path(root).joinpath(name, "processed")
        path1 = prep_path.joinpath(self._fields[0] + '.pt')
        if not (prep_path.exists() and prep_path.is_dir() and path1.exists()):
            # pre process and save
            mlog(f'first time preprocess {name}')
            raw_path = Path(root).joinpath(name, f"dgl_{name}.bin")
            graph = dgl.load_graphs(str(raw_path))[0][0]
            mlog(f'finish loading')
            # generate fake train idx
            num_train_nodes = int(graph.num_nodes() * 0.01)
            log_degs = torch.log(1+graph.in_degrees())
            probs = (log_degs / log_degs.sum()).numpy()
            train_idx = torch.from_numpy(np.random.choice(
                graph.num_nodes(), size=num_train_nodes, replace=False, p=probs)).long()
            mlog(f'finish generating train mask')
            mlog(graph)
            graph = dgl.to_bidirected(graph)
            mlog('finish to bidir:')
            mlog(graph)
            rowptr, col, _ = graph.adj_tensors(fmt='csc')
            mlog('finish get rowptr and col')
            # generate fake features and labels
            #x = torch.rand(graph.num_nodes(), fake_dim, dtype=torch.float16)
            #y = torch.randint(100,(graph.num_nodes(),)).squeeze().long()
            x = torch.Tensor()
            y = torch.Tensor()
            mlog('finish generating random features and labels')
            # construct and save
            meta_info = {'num classes': 100}
            split_idx = {'train': train_idx, 'val': torch.Tensor(), 'test': torch.Tensor()}
            dataset = self(name=name, x=x, y=y,
                   rowptr=rowptr, col=col,
                   split_idx=split_idx,
                   meta_info=meta_info)
            mlog(f'Saving processed data...')
            dataset.save(root, name)
            return dataset
        else:
            return self.from_dgl_path_if_exists(root, name, fake_dim)

    @classmethod
    def from_dgl_path_if_exists(self, _path, name, fake_dim):
        mlog(f'loading processed data')
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        assert path.exists() and path.is_dir()
        data = {
            field: torch.load(path.joinpath(field + '.pt'))
            for field in self._fields
        }
        if data['x'].shape[0] == 0:
            mlog(f'generating random features')
            num_nodes = data['rowptr'].shape[0]-1
            data['x'] = torch.rand(num_nodes, fake_dim, dtype=torch.float16)
            data['y'] = torch.randint(100,(num_nodes,)).squeeze().long()
            mlog(f'finish generating random features')
        else:
            data['y'] = data['y'].long()
            data['x'] = data['x'].to(torch.float16)
        assert data['name'] == name
        return self(**data)



    @classmethod
    def from_ogb(self, name: str, root='/home/data/os.environ["USER"]/datasets'):
        print('Obtaining dataset from ogb...')
        return self.process_ogb(PygNodePropPredDataset(name=name, root=root))

    @classmethod
    def process_ogb(self, dataset):
        print('Converting to fast dataset format...')
        data = dataset.data
        x = to_row_major(data.x).to(torch.float16)
        y = data.y.squeeze()

        if y.is_floating_point():
            y = y.nan_to_num_(-1)
            y = y.long()

        adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
        rowptr, col, _ = adj_t.to_symmetric().csr()
        return self(name=dataset.name, x=x, y=y,
                   rowptr=rowptr, col=col,
                   split_idx=dataset.get_idx_split(),
                   meta_info=dataset.meta_info.to_dict())

    @classmethod
    def from_path(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        path1 = path.joinpath(self._fields[0] + '.pt')
        if not (path.exists() and path.is_dir() and path1.exists()):
            print(f'First time preprocessing {name}; may take some time...')
            dataset = self.from_ogb(name, root=_path)
            print(f'Saving processed data...')
            dataset.save(_path, name)
            return dataset
        else:
            return self.from_path_if_exists(_path, name)

    @classmethod
    def from_path_if_exists(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        assert path.exists() and path.is_dir()
        data = {
            field: torch.load(path.joinpath(field + '.pt'))
            for field in self._fields
        }
        data['y'] = data['y'].long()
        data['x'] = data['x'].to(torch.float16)
        assert data['name'] == name
        return self(**data)

    def save(self, _path, name):
        path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
        path.mkdir(exist_ok=True)
        for i, field in enumerate(self._fields):
            torch.save(self[i], path.joinpath(field + '.pt'))

    def adj_t(self):
        num_nodes = self.x.size(0)
        return SparseTensor(rowptr=self.rowptr, col=self.col,
                            sparse_sizes=(num_nodes, num_nodes),
                            is_sorted=True, trust_data=True)

    def share_memory_(self):
        self.x.share_memory_()
        self.y.share_memory_()
        self.rowptr.share_memory_()
        self.col.share_memory_()

        for v in self.split_idx.values():
            v.share_memory_()

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_classes(self):
        return int(self.meta_info['num classes'])
