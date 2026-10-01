# Local Support Learning

<p align="center">

<a href="https://assafbk.github.io/website/">Assaf Ben-Kish</a>,
<a href="https://akarshkumar.com/">Akarsh Kumar</a>,
<a href="https://scholar.google.com/citations?user=pfGI-KcAAAAJ&hl=en">James Glass</a>,
<a href="https://scholar.google.com/citations?user=9aQUYVQAAAAJ&hl=iw">Raja Giryes</a>

<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"></a>
<a href="https://assafbk.github.io/lsl/"><img src="https://img.shields.io/badge/Project-Page-blue.svg"></a>

</p>
<br>

Continual fine-tuning with conditional LoRA adapters gated by GMM-based support estimation.

## Setup

Tested with the NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.10-py3` (PyTorch 2.9, CUDA 13.0) on a single GPU.

If you use the provided [`Dockerfile`](Dockerfile), everything is already installed and you only need to clone the repository (step 1). Otherwise:

### 1. Clone the repository

```bash
git clone https://github.com/assafbk/local_support_learning.git
cd local_support_learning
```

### 2. Set up a virtual environment

Using conda:

```bash
conda create -n lsl python=3.12
conda activate lsl
```

Using venv:

```bash
python3.12 -m venv lsl
source lsl/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # must come after torch is installed

# ifeval (lm-eval) needs the punkt tokenizer data
python -m nltk.downloader punkt_tab punkt
```

All models and datasets are downloaded from the Hugging Face Hub on first use.

## Training

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12929
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=<your-key>      # or set wandb_logger = False in configs/config_continual.py
export CODE_DIR=./local_support_learning
export HF_HOME=./hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LM_HARNESS_CACHE_PATH=./lm_eval_cache

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT train_continual.py
```

Optional: set `HF_HOME` to choose where models and datasets are cached, and `LM_HARNESS_CACHE_PATH` to choose where lm-eval caches its requests.

## Configuration

The configuration file is [`configs/config_continual.py`](configs/config_continual.py).

General configurations:

- `model_name`: Any Qwen2.5-Instruct model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`). Initialized to `Qwen/Qwen2.5-7B-Instruct`.
- `continual_learning_datasets`: Tasks to learn, in order. Any combination of `chemdata`, `igbo_translation`, `cybersecurity_ift`.
- `activate_cl_method`: `False` runs a plain LoRA baseline (no gating). Initialized to `True`.
Method configurations:

- `lora_r`: LoRA rank per phase.
- `cl_memory_type`: Support estimator used for gating. Initialized to `"gmm"`.
- `activate_jl`: Project inputs to `jl_k` dims (Johnson-Lindenstrauss) before fitting and gating. Initialized to `True`.
- `jl_k`: JL target dimension.
- `flush_cl_memory_after_each_epoch`: Refit the task's support at the end of every epoch from the updated representations. Initialized to `True`.
- `smoothing_alpha`: Gate smoothing across tokens at inference, between 0 and 1. Lower values smooth more, and `1.0` disables smoothing.

GMM configurations:

- `gmm_n_pos_comps` / `gmm_n_neg_comps`: GMM components for the task (positive) / generic (negative) data.
- `gmm_fit_num_tokens`: Number of tokens used to fit each GMM.
- `gmm_skip_on_fail`: If a module's GMM fit fails, disable that module's LoRA instead of crashing. Initialized to `True`.
- `train_neg_gmm`: Fit the negative GMMs in this run. `False` loads them from `gmm_neg_path`. Initialized to `True`.
- `gmm_neg_path`: Saved negative GMMs from a previous run (used when `train_neg_gmm=False`).
- `use_prepared_negatives_set`: Fit the negative GMMs on a prepared 1M-token sample. `False` uses FineWeb-Edu 10BT (~40GB download). Initialized to `True`.

Evaluation configurations:

- `lm_eval_tasks`: Benchmarks for measuring forgetting. Initialized to `humaneval, gsm8k, ifeval`.
- `skip_lm_eval`: Skip the benchmarks (cached base-model scores are still used). Initialized to `False`.
- `lm_eval_compute_last_epoch_only`: Run the benchmarks only after the last epoch of each task. Initialized to `False`.

Save / load configurations:

- `save_cl_state`: Save the trained adapters and GMMs after every epoch, to `output/<run>/cl_state_<task>_epoch_<epoch>.pt`. Initialized to `False`.
- `load_cl_state_path`: Path to a saved `cl_state_*.pt` file to load instead of training from scratch. Initialized to `''` (no loading).

Check out [`configs/config_continual.py`](configs/config_continual.py) for more configurations.

## Outputs

- `output/<timestamp>_<run_name>/` holds the fitted GMMs and the per-sample lm-eval generations.
- `results/<timestamp>_<run_name>_best/` holds the final continual-learning summary table (CSV).
- `base_model_performance_cache/` caches the base model's zero-shot lm-eval scores, so later runs reuse them.

## Citation

```bibtex
@article{benkish2026lsl,
  title   = {Local Support Learning},
  author  = {Ben-Kish, Assaf and Kumar, Akarsh and
             Glass, James and Giryes, Raja},
  journal = {arXiv preprint},
  year    = {2026}
}
```
