# ddp_setup.py (or top of your train.py)
import os, torch, torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import contextmanager
from datetime import timedelta


use_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup_ddp(model, train_dl, test_dl):
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return model, train_dl, test_dl
    model  = wrap_model_ddp(model)
    train_dl = make_ddp_dataloader(train_dl.dataset, train_dl.batch_size)
    test_dl  = make_ddp_dataloader(test_dl.dataset, test_dl.batch_size)
    return model, train_dl, test_dl

@contextmanager
def dist_group(backend="nccl", **kwargs):
    if not use_ddp:
        yield
        return

    assert torch.cuda.is_available(), "CUDA is not available. DDP requires GPUs."
    assert "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=20), **kwargs)
    try:
        yield
    except Exception as e:
        print(f"Error in dist_group:\n{e}")
        raise e
    finally:
        print(f"Destroying process group on rank {rank}")
        dist.destroy_process_group()

def wrap_model_ddp(model):
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized. Please launch the script with torchrun."
    # enable DDP
    model.to(torch.cuda.current_device())
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        output_device=torch.cuda.current_device(),
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    model.config = model.module.config  # to access config from model
    model.tie_weights = lambda: None  # disable tie_weights call in HFLM
    model.generate = model.module.generate  # to access generate from model

    return model

def make_ddp_dataloader(dataset, batch_size, num_workers=0):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=True,              # avoids last short batch desync
                      pin_memory=False,
                      num_workers=num_workers,
                      sampler=DistributedSampler(dataset, shuffle=False),
                      persistent_workers=False,
                      )

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def all_reduce(obj, normalize=False):
    if not torch.distributed.is_initialized():
        return obj
    world_size = torch.distributed.get_world_size()
    if isinstance(obj, torch.Tensor):
        torch.distributed.all_reduce(obj, op=torch.distributed.ReduceOp.SUM)
        return obj / world_size if normalize else obj
    elif isinstance(obj, (list, tuple)):
        obj = type(obj)(all_reduce(o) for o in obj)
        return obj
    elif isinstance(obj, dict):
        obj = {k: all_reduce(v) for k, v in obj.items()}
        return obj
    elif isinstance(obj, (int, float)):
        t = torch.tensor(obj, device="cuda")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return (t / world_size if normalize else t).item()

def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_gather_list(items):
    """All-gather a list of arbitrary picklable objects (e.g. decoded strings) from every rank.

    Returns every rank's items concatenated in rank order, so the result is identical and deterministically
    ordered on all ranks. Corpus-level metrics (sacreBLEU) need the whole hypothesis/reference set on one
    rank rather than a per-rank sum, which is what all_reduce gives. Single rank / no DDP -> unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return items
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, items)
    return [item for rank_items in gathered for item in rank_items]


def all_gather_var(tensor):
    """All-gather a 2D tensor whose first dim (row count) differs across ranks.

    Returns the rows from every rank concatenated in rank order (rank 0 first, then
    rank 1, ...), so the result is identical and deterministically ordered on all
    ranks. Single rank / no DDP -> returns the input unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    world_size = dist.get_world_size()
    device = tensor.device

    # 1) exchange row counts so every rank knows the gathered shapes.
    n = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    counts = [torch.zeros_like(n) for _ in range(world_size)]
    dist.all_gather(counts, n)
    counts = [int(c.item()) for c in counts]
    max_n = max(counts)

    # 2) pad each rank's tensor to max_n rows, all-gather the fixed-size buffers, trim.
    feat = tensor.shape[1]
    padded = tensor.new_zeros((max_n, feat))
    padded[: tensor.shape[0]] = tensor
    bufs = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(bufs, padded)
    return torch.cat([bufs[r][: counts[r]] for r in range(world_size)], dim=0)


def broadcast_var(tensor, feat, dtype, src=0, device=None):
    """Broadcast a 2D tensor of unknown row count from ``src`` to all ranks (NCCL, on-GPU).

    ``src`` passes its tensor; non-src ranks pass None. The row count is broadcast first
    (so receivers can allocate), then the rows. ``feat``/``dtype`` describe the columns so
    non-src ranks can build the receive buffer. Single rank / no DDP -> returns ``tensor``.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    if device is None:
        device = tensor.device if tensor is not None else torch.cuda.current_device()

    # 1) broadcast the row count so non-src ranks know how big to allocate.
    n = torch.tensor([tensor.shape[0] if tensor is not None else 0], device=device, dtype=torch.long)
    dist.broadcast(n, src=src)
    n = int(n.item())

    # 2) allocate on non-src ranks, then broadcast the rows.
    if tensor is None:
        tensor = torch.zeros((n, feat), device=device, dtype=dtype)
    dist.broadcast(tensor, src=src)
    return tensor