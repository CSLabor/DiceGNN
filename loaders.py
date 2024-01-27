from DiceGNN.utils import get_logger
mlog = get_logger()

from third_party.salient.driver.dataset import FastDataset
import dgl
import math
import torch

def load_shared_data(dataset_name, dataset_root, shared_graph=False):
    """
    graph topo & feature data can be shared between SALIENT and DGL
    """
    assert dataset_name in ['ogbn-papers100M', 'twitter', 'uk']
    if dataset_name in ['twitter', 'uk']:
        dataset = FastDataset.from_dgl(name=dataset_name, root=dataset_root)
    else:
        dataset = FastDataset.from_path(dataset_root, dataset_name)

    train_idx = dataset.split_idx['train']
    num_classes = dataset.num_classes

    x = dataset.x
    y = dataset.y.unsqueeze(-1)
    row = dataset.rowptr
    col = dataset.col

    # in-place pinning of four arrays
    cudart = torch.cuda.cudart()
    torch.cuda.check_error(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0))
    torch.cuda.check_error(cudart.cudaHostRegister(y.data_ptr(), y.numel() * y.element_size(), 0))
    torch.cuda.check_error(cudart.cudaHostRegister(row.data_ptr(), row.numel() * row.element_size(), 0))
    torch.cuda.check_error(cudart.cudaHostRegister(col.data_ptr(), col.numel() * col.element_size(), 0))

    graph_shm = None
    if shared_graph:
        # prepare graph in shared memory for multiprocessing
        row_copy = row.clone().detach()
        col_copy = col.clone().detach()
        row_copy.share_memory_()
        col_copy.share_memory_()
        graph_shm = dgl.graph(('csc', (row_copy, col_copy, torch.Tensor())))
        graph_shm = graph_shm.formats('csc')

    # torch.cuda.check_error(cudart.cudaHostUnregister(row.data_ptr())) 
    mlog('finish loading shared data')
    return x, y, row, col, graph_shm, train_idx, num_classes

def partition_train_idx(all_train_idx, ratio=0.5):
    """
    return: CPU_train_idx, GPU_train_idx
    """
    temp_train_idx = all_train_idx[torch.randperm(all_train_idx.shape[0])]
    sep = int(all_train_idx.shape[0] * ratio)
    mlog(f"split into two part, salient {sep} : dgl {all_train_idx.shape[0]-sep}")
    return temp_train_idx[:sep], temp_train_idx[sep:].cuda()
