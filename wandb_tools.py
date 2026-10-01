import sys, signal, time
from contextlib import contextmanager, nullcontext
from loguru import logger
import wandb
import numpy as np


def _on_signal(signum, frame):
    logger.error(f"Received signal {signum}; shutting down gracefully")
    try:
        if wandb.run: wandb.finish()
    except Exception as e:
        logger.error(f"W&B finish failed during signal: {e}")
    sys.exit(128 + signum)

for s in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
    signal.signal(s, _on_signal)


def _safe_wandb_log(*args, **kwargs):
    while True:
        try:
            wandb.log(*args, **kwargs)
            break
        except Exception as e:
            logger.error(f"W&B log failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def wandb_log(*args, **kwargs):
    assert "commit" not in kwargs.keys()
    _safe_wandb_log(*args, commit=False, **kwargs)

def wandb_commit():
    _safe_wandb_log({}, commit=True)
    

@contextmanager
def wandb_init(config, local_rank):
    if local_rank > 0:
        yield nullcontext()
    else:
        with wandb.init(
            name=f'{config.wandb_run_name}',
            group=None,
            project=getattr(config, 'wandb_project', 'unnamed'),
            entity=None,
            mode="online" if config.wandb_logger else "disabled",
            config=config,
            dir=config.logsdir,
            settings=wandb.Settings(code_dir=config.logsdir),  # saves all files in current directory
        ) as run:
            try:
                yield run
            except Exception as e:
                logger.error(f"Error during training:\n{e}")
                raise e


def log_head_hists_wandb(
    x,                     # (B, H, D) torch.Tensor or np.ndarray
    name: str = "-",
    bins: int = 100,
    shared_bins: bool = True,
    every: int = 1_000,      # log every N steps
    max_per_head: int | None = 200_000,  # subsample to cap payload
):
    if wandb.run.step % every != 0:
        return

    # to numpy
    x_np = x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
    assert x_np.ndim == 3
    B, H, D = x_np.shape

    # W&B cap
    MAX_BINS = 512
    eff_bins = min(int(bins), MAX_BINS)

    # shared bin range (optional)
    hist_range = None
    if shared_bins:
        flat = x_np.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            lo, hi = float(np.min(flat)), float(np.max(flat))
            eps = (hi - lo) * 1e-6 + 1e-12
            hist_range = (lo - eps, hi + eps)

    logs = {}
    rng = np.random.default_rng(0)
    for h in range(H):
        vals = x_np[:, h, :].ravel()
        vals = vals[np.isfinite(vals)]
        if max_per_head and vals.size > max_per_head:
            idx = rng.choice(vals.size, size=max_per_head, replace=False)
            vals = vals[idx]

        # compute with eff_bins ≤ 512
        counts, edges = np.histogram(vals, bins=eff_bins, range=hist_range)
        logs[f"{name}/_{h:02d}"] = wandb.Histogram(np_histogram=(counts, edges))

    wandb_log(logs)
