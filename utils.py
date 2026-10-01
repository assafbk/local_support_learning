import torch
from tabulate import tabulate
from loguru import logger
import numpy as np
import os
import csv

'''
Calculates gradient norms according to two groups of parameters: layers and other (embedding matrices, etc.).
We dont sqrt total_reg_norms, total_other_norms because we want to see their relative value w.r.t the total grad (this way they sum to grad_norm_sqrd)
We sqrt the values of reg_norms because we just want the correct scale, no need for relative value w.r.t all other params.
'''
def get_grad_norms(model, device='cuda'):
    layer_norms = torch.zeros(model.config.num_hidden_layers, device=device)
    total_layer_norms = 0
    total_other_norms = 0
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        if 'layers.' not in name:
            grad_sq = param.grad.norm() ** 2
            total_other_norms += grad_sq
        
        else:
            layer_idx = int(name.split('layers.')[1].split('.')[0])
            grad_sq = param.grad.norm() ** 2
            
            layer_norms[layer_idx] += grad_sq
            total_layer_norms += grad_sq
    
    return total_layer_norms, total_other_norms, torch.sqrt(layer_norms)

'''
Calculates the average Frobenius norm of MLP weight matrices per layer, split into predictive and regular layers.
Returns two tensors of shape (num_hidden_layers,): one for pred layers, one for regular layers.
Each entry is the mean Frobenius norm of the 3 MLP matrices (gate_proj, up_proj, down_proj) in that layer.
'''
@torch.no_grad()
def get_mlp_frobenius_norms(model, device='cuda'):
    pred_frob = torch.zeros(model.config.num_hidden_layers, device=device)
    reg_frob = torch.zeros(model.config.num_hidden_layers, device=device)
    pred_count = torch.zeros(model.config.num_hidden_layers, device=device)
    reg_count = torch.zeros(model.config.num_hidden_layers, device=device)

    for name, param in model.named_parameters():
        if 'layers.' not in name or 'mlp.' not in name:
            continue
        if not name.endswith('.weight'):
            continue

        layer_idx = int(name.split('layers.')[1].split('.')[0])
        frob = param.data.norm()

        if 'predictive' in name:
            pred_frob[layer_idx] += frob
            pred_count[layer_idx] += 1
        else:
            reg_frob[layer_idx] += frob
            reg_count[layer_idx] += 1

    pred_frob = pred_frob / pred_count.clamp(min=1)
    reg_frob = reg_frob / reg_count.clamp(min=1)

    return pred_frob, reg_frob

'''
Calculates the amount of trainable parameters of two param groups: layers and other (embedding matrices, etc.).
'''
def get_param_count(model):
    
    total_layers_params = 0
    total_other_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue
        
        if 'layers.' not in name:
            total_other_params += param.numel()
        
        else:
            total_layers_params += param.numel()
    
    return total_layers_params, total_other_params


"""Save per-seed finetune eval results as CSVs (PPL rows then R² rows)."""
def log_finetune_eval_summary(finetune_datasets, ppl_results, task_eval_results, lm_eval_results, pretrain_ppl, save_dir):
    """
    Log summary of finetune (transfer) evaluation as two tables, one row per metric.
    Columns: zero-shot + per-epoch (each dataset is an independent finetune run).
    Table 1 (scores): per-dataset task accuracy + lm_eval tasks.
    Table 2 (ppl): per-dataset eval PPL.
    ppl_results / task_eval_results: {dataset: [zero_shot, epoch0, epoch1, ...]}
    lm_eval_results: {dataset: {"lm_eval_task": [zero_shot, epoch0, ...], ..., "total": [...]}}
    """
    max_epochs = max(len(ppl_results[ds]) for ds in finetune_datasets)  # includes the zero-shot at index 0
    headers = [""] + ["zero-shot"] + [f"epoch {i}" for i in range(max_epochs - 1)]

    def row(label, vals, fmt):
        return [label] + [fmt(v) for v in vals] + [""] * (max_epochs - len(vals))

    # Table 1: task accuracy + lm_eval scores
    score_rows = []
    for ds in finetune_datasets:
        score_rows.append(row(f"{ds} Eval Acc", task_eval_results[ds], lambda v: f"{v:.4f}"))
    for ds in finetune_datasets:
        for lm_label, vals in lm_eval_results[ds].items():
            score_rows.append(row(f"{ds} lm_eval {lm_label}", vals, lambda v: f"{v:.4f}"))

    # Table 2: ppl
    ppl_rows = [row(f"{ds} Eval PPL", ppl_results[ds], lambda v: f"{v:.2f}") for ds in finetune_datasets]

    logger.info(f"\n{'='*60}")
    logger.info(f"[FINETUNE EVAL SUMMARY] (pretrain PPL={pretrain_ppl:.2f})")
    logger.info(f"\n{tabulate(score_rows, headers=headers, tablefmt='simple')}")
    logger.info(f"\n{tabulate(ppl_rows, headers=headers, tablefmt='simple')}")
    logger.info(f"{'='*60}")
    finetune_log_results_csv_format(headers, score_rows, ppl_rows, save_dir)

"""Log finetune results in CSV format and save to save_dir."""
def finetune_log_results_csv_format(headers, score_rows, ppl_rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "finetune_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(score_rows + ppl_rows)
    logger.info(f"[FT CSV] Saved results to {csv_path}")


def log_continual_learning_eval_summary(cl_datasets, ppl_results, task_eval_results, lm_eval_results, pretrain_ppl, save_dir):
    """
    Log summary of continual learning evaluation as two tables, one row per metric.
    Columns: training epochs across all tasks sequentially.
    Table 1 (scores): per-dataset task accuracy + lm_eval tasks.
    Table 2 (ppl): per-dataset eval PPL.
    ppl_results / task_eval_results: {(task_idx, eval_dataset_name): [zero_shot, epoch0, epoch1, ...]}
    lm_eval_results: {task_idx: {"lm_eval_task": [zero_shot, epoch0, ...], ..., "total": [...]}}
    """
    epochs_per_task = [len(ppl_results.get((task_idx, train_ds), [])) for task_idx, train_ds in enumerate(cl_datasets)]

    # lm_eval task labels (consistent across tasks), taken from the first available entry
    lm_eval_labels = list(lm_eval_results[0].keys()) if 0 in lm_eval_results else []

    # Build multi-level header: group epochs by training dataset
    headers = [""]
    for task_idx, train_ds in enumerate(cl_datasets):
        for ep in range(epochs_per_task[task_idx]):
            headers.append(f"{train_ds} ep{ep}")

    def cells(get_vals, fmt):
        # one row of per-(task_idx, epoch) cells, formatted with fmt
        out = []
        for task_idx in range(len(cl_datasets)):
            vals = get_vals(task_idx)
            for ep in range(epochs_per_task[task_idx]):
                out.append(fmt(vals[ep]) if ep < len(vals) else "")
        return out

    # Table 1: task accuracy + lm_eval scores
    score_rows = []
    for eval_ds in cl_datasets:
        score_rows.append([f"{eval_ds} Eval Acc"] + cells(lambda ti: task_eval_results.get((ti, eval_ds), []), lambda v: f"{v:.4f}"))
    score_rows.append(["total Eval Acc"] + cells(lambda ti: task_eval_results.get((ti, 'total'), []), lambda v: f"{v:.4f}"))  # mean over datasets per (task, epoch)
    for lm_label in lm_eval_labels:
        score_rows.append([f"lm_eval {lm_label}"] + cells(lambda ti: lm_eval_results.get(ti, {}).get(lm_label, []), lambda v: f"{v:.4f}"))

    # Table 2: ppl
    ppl_rows = []
    for eval_ds in cl_datasets:
        ppl_rows.append([f"{eval_ds} Eval PPL"] + cells(lambda ti: ppl_results.get((ti, eval_ds), []), lambda v: f"{v:.2f}"))

    logger.info(f"\n{'='*60}")
    logger.info(f"[CONTINUAL LEARNING EVAL SUMMARY] (pretrain PPL={pretrain_ppl:.2f})")
    logger.info(f"  Task order: {' -> '.join(cl_datasets)}")
    logger.info(f"\n{tabulate(score_rows, headers=headers, tablefmt='simple')}")
    logger.info(f"\n{tabulate(ppl_rows, headers=headers, tablefmt='simple')}")
    logger.info(f"{'='*60}")
    cl_log_results_csv_format(headers, score_rows, ppl_rows, save_dir)

"""Log CL results in CSV format and save to save_dir."""
def cl_log_results_csv_format(headers, score_rows, ppl_rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "cl_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(score_rows + ppl_rows)
    logger.info(f"[CL CSV] Saved results to {csv_path}")

'''3D voxel plot of the feature map. Occupied cells are colored by fitness (viridis); empty cells are transparent.'''
def visualize_feature_map(database, num_bins, save_dir=None):
    from map_elites import TRANSFER_LOG_MIN,TRANSFER_LOG_MAX, TRANSFER_MILESTONE

    occupied = torch.zeros(num_bins, num_bins, num_bins, dtype=torch.bool)
    fitness_grid = torch.zeros(num_bins, num_bins, num_bins)
    for (cl_b, tr_b, rob_b), (_, fitness, _, _) in database.items():
        occupied[cl_b, tr_b, rob_b] = True
        fitness_grid[cl_b, tr_b, rob_b] = fitness

    cmap = plt.get_cmap("viridis")
    fmin = float(fitness_grid[occupied].min()) if occupied.any() else 0.0
    fmax = float(fitness_grid[occupied].max()) if occupied.any() else 1.0
    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    colors = cmap(norm(fitness_grid.numpy()))
    colors[..., 3] = occupied.numpy().astype(float)  # alpha = 1 occupied, 0 empty

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occupied.numpy(), facecolors=colors, edgecolor="k", linewidth=0.2)

    # force the full grid extent on every axis (not just occupied cells)
    ax.set_xlim(0, num_bins)
    ax.set_ylim(0, num_bins)
    ax.set_zlim(0, num_bins)

    # label ticks with the lower-edge bin value, not the bin index
    tick_idx = list(range(num_bins + 1))
    cl_vals  = [f"{i / num_bins:.2f}" for i in tick_idx]
    rob_vals = [f"{i / num_bins:.2f}" for i in tick_idx]
    tr_vals  = [f"{int(round(10 ** (TRANSFER_LOG_MAX - (i / num_bins) * (TRANSFER_LOG_MAX - TRANSFER_LOG_MIN))))}" for i in tick_idx]
    ax.set_xticks(tick_idx); ax.set_xticklabels(cl_vals, fontsize=7)
    ax.set_yticks(tick_idx); ax.set_yticklabels(tr_vals, fontsize=7)
    ax.set_zticks(tick_idx); ax.set_zticklabels(rob_vals, fontsize=7)

    ax.set_xlabel("cl accuracy")
    ax.set_ylabel("transfer samples")
    ax.set_zlabel("robustness accuracy")
    ax.set_title(f"map-elites feature map ({int(occupied.sum())}/{num_bins**3} cells)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="fitness", shrink=0.6)

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "feature_map_3d.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig

'''Plotly version of the feature map: renders interactively in a notebook (drag to rotate, scroll to zoom). Each occupied cell becomes a colored cube via Mesh3d; empty cells are simply not drawn.'''
def visualize_feature_map_plotly(database, num_bins):
    import plotly.graph_objects as go
    from map_elites import TRANSFER_LOG_MIN,TRANSFER_LOG_MAX, TRANSFER_MILESTONE

    cl_axis_vals  = [i / num_bins for i in range(num_bins + 1)]
    rob_axis_vals = [i / num_bins for i in range(num_bins + 1)]
    tr_axis_vals  = [int(round(10 ** (TRANSFER_LOG_MAX - (i / num_bins) * (TRANSFER_LOG_MAX - TRANSFER_LOG_MIN)))) for i in range(num_bins + 1)]

    fitnesses = [f for (_, f, _, _) in database.values()]
    fmin, fmax = (min(fitnesses), max(fitnesses)) if fitnesses else (0.0, 1.0)

    meshes = []
    for (cl_b, tr_b, rob_b), (_, fitness, _, _) in database.items():
        # 8 corners of the unit cube at (cl_b, tr_b, rob_b)
        x = [cl_b, cl_b + 1, cl_b + 1, cl_b, cl_b, cl_b + 1, cl_b + 1, cl_b]
        y = [tr_b, tr_b, tr_b + 1, tr_b + 1, tr_b, tr_b, tr_b + 1, tr_b + 1]
        z = [rob_b, rob_b, rob_b, rob_b, rob_b + 1, rob_b + 1, rob_b + 1, rob_b + 1]
        # 12 triangles (2 per face) using the 8 corner indices
        i = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
        j = [1, 2, 4, 3, 5, 6, 5, 6, 6, 7, 7, 4]
        k = [2, 3, 7, 7, 6, 7, 6, 2, 7, 3, 4, 0]
        meshes.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=[fitness] * 8, cmin=fmin, cmax=fmax,
            colorscale="Viridis", showscale=False,
            flatshading=True, opacity=1.0,
            hovertext=f"cl={cl_axis_vals[cl_b]:.2f} tr={tr_axis_vals[tr_b]} rob={rob_axis_vals[rob_b]:.2f} fitness={fitness:.3f}",
            hoverinfo="text",
        ))

    # one invisible mesh just to draw the colorbar
    meshes.append(go.Mesh3d(
        x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0], i=[0], j=[1], k=[2],
        intensity=[fmin, fmax, (fmin + fmax) / 2],
        cmin=fmin, cmax=fmax, colorscale="Viridis",
        showscale=True, colorbar=dict(title="fitness"),
        opacity=0,
    ))

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title=f"map-elites feature map ({len(database)}/{num_bins**3} cells)",
        scene=dict(
            xaxis=dict(title="cl accuracy",        tickvals=list(range(num_bins + 1)), ticktext=[f"{v:.2f}" for v in cl_axis_vals],  range=[0, num_bins]),
            yaxis=dict(title="transfer samples",   tickvals=list(range(num_bins + 1)), ticktext=[str(v) for v in tr_axis_vals],      range=[0, num_bins]),
            zaxis=dict(title="robustness accuracy",tickvals=list(range(num_bins + 1)), ticktext=[f"{v:.2f}" for v in rob_axis_vals], range=[0, num_bins]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


class TokenizeAndPack(torch.utils.data.IterableDataset):
    def __init__(self, base_ds, tokenizer, seq_len, text_field="text"):
        self.ds, self.tok, self.T = base_ds, tokenizer, seq_len
        self.text_field, self.eos = text_field, tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        buf, src, sid = [], [], 0
        for ex in self.ds:
            ids = self.tok(ex[self.text_field], add_special_tokens=False)["input_ids"]
            if self.eos is not None and (not ids or ids[-1] != self.eos): ids.append(self.eos)
            sid += 1
            buf.extend(ids); src.extend([sid]*len(ids))
            while len(buf) >= self.T:
                out = torch.tensor(buf[:self.T], dtype=torch.long)
                s   = src[:self.T]
                num = 1 + sum(s[i] != s[i-1] for i in range(1, self.T))
                yield {"input_ids": out, "num_samples": num}
                del buf[:self.T]; del src[:self.T]