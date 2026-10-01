import os
from dataclasses import dataclass, field
from typing import Optional


def _env_float(name, default, scale=1.0):
    v = os.environ.get(name)
    return (float(v) if v is not None and v != '' else default) * scale


def _env_int(name, default, scale=1):
    v = os.environ.get(name)
    base = int(float(v)) if v is not None and v != '' else default  # via float so scientific notation ('1e7') parses
    return base * scale

def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None or v == '':
        return default
    return v.strip().lower() in ('1', 'true', 't', 'yes', 'y')


@dataclass
class Configuration:
    
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    run_cl_eval: bool = True
    continual_learning_datasets: list = field(default_factory=lambda: ['igbo_translation']) # igbo_translation / chemdata / cybersecurity_ift / any combination
    
    ft_global_batch_size: int = field(default_factory=lambda: _env_int('FT_GLOBAL_BATCH_SIZE', 32))
    ft_warmup_ratio: float = 0.1
    ft_decay_ratio: float = 0.1
    ft_stable_ratio: float = 1 - ft_decay_ratio - ft_warmup_ratio
    ft_max_num_epochs: int = 1
    ft_epochs_per_dataset: dict = field(default_factory=lambda: {'igbo_translation': 3, 'chemdata': 3, 'cybersecurity_ift':1}) # Per-dataset epoch overrides, e.g. {'igbo_translation': 2}. Datasets absent here use ft_max_num_epochs.
    ft_learning_rate: float = field(default_factory=lambda: _env_float('FT_LEARNING_RATE', 1e-4))
    ft_min_learning_rate: float = field(default_factory=lambda: _env_float('FT_LEARNING_RATE', 1e-4, scale=0.01)) # ft_learning_rate / 100

    # Training parameters
    local_batch_size: int = field(default_factory=lambda: _env_int('FT_GLOBAL_BATCH_SIZE', 32))
    seq_len_train: int = 1024
    use_grad_checkpointing: bool = True
        
    # Evaluation parameters
    eval_batch_size: int = field(default_factory=lambda: _env_int('FT_GLOBAL_BATCH_SIZE', 32))
    lm_eval_temperature: float = 1.0
    lm_eval_top_p: float = 0.95
    seq_len_eval: int = 1024

    lm_eval_tasks: list = field(default_factory=lambda: ["humaneval", "gsm8k", "ifeval"])
    skip_lm_eval: bool = False
    lm_eval_compute_last_epoch_only: bool = False
    lm_eval_limit: Optional[float] = field(default_factory=lambda: float(os.environ['LM_EVAL_LIMIT']) if os.environ.get('LM_EVAL_LIMIT') else None)
    base_model_performance_cache_dir: str = "base_model_performance_cache"  # cache the base model's zero-shot lm-eval scores + captured generations

    # Optimizer parameters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # cl method parameters
    activate_cl_method: bool = field(default_factory=lambda: _env_bool('ACTIVATE_CL', True))
    use_conditional_lora: bool = True
    allocate_lora_per_phase: bool = True
    lora_r: int = field(default_factory=lambda: _env_int('LORA_R', 128))
    lora_alpha: float = field(default_factory=lambda: _env_int('LORA_R', 128, scale=2)) #16.0

    # cl memory backend
    cl_memory_type: str = 'gmm'
    activate_jl: bool = field(default_factory=lambda: _env_bool('ACTIVATE_JL', True))  # Johnson-Lindenstrauss: project inputs to jl_k dims before storing/gating cl_memory
    jl_k: int = field(default_factory=lambda: _env_int('JL_K', 256))  # JL target dim (used only when activate_jl)
    flush_cl_memory_after_each_epoch: bool = True  # empty the accumulating LoRA's cl_memory at the end of each epoch except the last (rebuild the support from the updated representations)
    smoothing_alpha: float = field(default_factory=lambda: _env_float('SMOOTHING_ALPHA', 0.2))

    ## gmms
    gmm_n_pos_comps: int = field(default_factory=lambda: _env_int('GMM_N_POS_COMPS', 16))   # components in the positive (in-task) GMM
    gmm_n_neg_comps: int = field(default_factory=lambda: _env_int('GMM_N_NEG_COMPS', 32))   # components in the negative (generic corpus) GMM
    gmm_covariance_type: str = 'diag'
    gmm_seed: int = 0
    gmm_fit_num_tokens: int = field(default_factory=lambda: _env_int('GMM_FIT_NUM_TOKENS', int(float(1e6))))
    gmm_skip_on_fail: bool = True  # if a module's GMM fit raises and error, disable that module's LoRA and continue instead of crashing the run
    train_neg_gmm: bool = True  # True: fit the negative GMMs on the negative corpus and save them under the run's output dir. False: load them from gmm_neg_path.
    gmm_neg_path: str = ''  # LOAD source for the negative GMMs (train_neg_gmm=False). Points at a previous run's file; never written to.
    num_samples_to_save_per_pt_task: int = 200
    use_prepared_negatives_set: bool = True # use the prepared 1M token sample. If set to False will use fineweb_edu 10BT split, which requires 40GB to download.

    # cl state save / load (adapters + cl_memory). For debug: seamlessly reload a trained CL model.
    save_cl_state: bool = False        # save the cl state after every epoch's evals
    load_cl_state_path: str = ''

    # System
    wandb_project: str = 'continual_learning'
    logsdir: str = "./logs"
    seed: int = field(default_factory=lambda: _env_int('SEED', 123))
    wandb_logger: bool = True
    logger_step_size: int = 1
    wandb_run_name: str = ''  # set in __post_init__ so it reflects env-overridden values (e.g. FT_LEARNING_RATE)

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split('/')[-1]
            lora_allocation_str = f'lora_per_phase_r_{self.lora_r}_' if self.allocate_lora_per_phase else f'single_lora_r_{self.lora_r}_' if self.use_conditional_lora else 'no_lora_'
            cl_method_str = f'cl_method_JL_k_{self.jl_k}_gmms_nc_pos_{self.gmm_n_pos_comps}_nc_neg_{self.gmm_n_neg_comps}_sm_alpha_{self.smoothing_alpha}' if self.activate_cl_method else f''
            run_name_addon = f'' + lora_allocation_str + cl_method_str
            datasets_str = '_'.join(self.continual_learning_datasets)
            self.wandb_run_name = f'{model_short}_{datasets_str}_seed_{self.seed}_bs_{self.ft_global_batch_size}_lr_{self.ft_learning_rate}_{run_name_addon}'