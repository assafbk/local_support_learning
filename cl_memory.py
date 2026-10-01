from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import Normal


# pomegranate computes diagonal covariances as E[x^2] - E[x]^2 in from_summaries() and never applies
# its own min_cov floor there (unlike sklearn's reg_covar, which is added every M-step). Sometimes some
# components collapse to a variance at the fp32 cancellation floor, and E[x^2]-E[x]^2 rounds slightly
# NEGATIVE -> the "Variances must be positive." error. _reset_cache runs the covs<0 check, so we floor
# covs at the top of _reset_cache (before that check) -- restoring the reg_covar behavior pomegranate
# omits. Patched once at import; idempotent across re-imports.
GMM_MIN_COV = 1e-6

if not getattr(Normal, "_reset_cache_floored", False):
    _orig_reset_cache = Normal._reset_cache

    def _floored_reset_cache(self):
        if getattr(self, "covs", None) is not None and torch.is_tensor(self.covs) and self.covs.numel():
            with torch.no_grad():
                self.covs.clamp_(min=GMM_MIN_COV)
        return _orig_reset_cache(self)

    Normal._reset_cache = _floored_reset_cache
    Normal._reset_cache_floored = True


class GMMClassifierGPU():
    """A per-LoRA membership test: ``predict(X) > 0`` means "inside this LoRA's memory".

    The two GMMs are fitted at different times (the negative once at the start of training, the
    positive at the end of every epoch), so ``fit_pos`` / ``fit_neg`` exist alongside the combined
    ``fit``. Callers should treat the pos/neg split as an implementation detail and go through
    ``predict``.
    """

    def __init__(self, pos_kwargs, neg_kwargs, device="cuda"):
        self.pos_kwargs = dict(pos_kwargs)
        self.neg_kwargs = dict(neg_kwargs)
        self.device = device
        # pomegranate's tol is on the TOTAL log-likelihood improvement, so it has to scale with the
        # number of points or EM never converges early and always burns max_iter.
        self.tol_ratio = 0.1
        self.pos_gmm = self.init_gmm(self.pos_kwargs, device)
        self.neg_gmm = self.init_gmm(self.neg_kwargs, device)
        # Stacked params for the batched predict. pomegranate scores components in a Python loop - too slow for generation.
        self.params_vectorized = {"pos": None, "neg": None} # {"pos"/"neg": (log_priors (K,), means (K,d), covs (K,d))}; None until that GMM is fitted.

    @staticmethod
    def init_gmm(kwargs, device="cuda"):
        return GeneralMixtureModel(
            [Normal(covariance_type=kwargs['covariance_type']) for _ in range(kwargs['n_components'])],
            random_state=kwargs['random_state'], max_iter=100,
            ).to(device)

    @property
    def is_fitted(self) -> bool:
        """True once both GMMs have been fitted (i.e. predict has params to use). Callers gate on this."""
        return self.params_vectorized["pos"] is not None and self.params_vectorized["neg"] is not None

    def reset_pos(self) -> None:
        """Drop the positive fit so it can be rebuilt from scratch (see flush_cl_memory). The
        negative GMM describes a fixed corpus and is fitted once per run, so it survives."""
        self.pos_gmm = self.init_gmm(self.pos_kwargs, self.device)
        self.params_vectorized["pos"] = None
        assert self.is_fitted == False

    def fit(self, X_pos, X_neg):
        self.fit_pos(X_pos)
        self.fit_neg(X_neg)

    def fit_pos(self, X_pos):
        self.pos_gmm.tol = self.tol_ratio * X_pos.shape[0]
        self.pos_gmm.fit(X_pos)
        self._assert_finite_gmm(self.pos_gmm)
        self.params_vectorized["pos"] = self._vectorize_params(self.pos_gmm)

    def fit_neg(self, X_neg):
        self.neg_gmm.tol = self.tol_ratio * X_neg.shape[0]
        self.neg_gmm.fit(X_neg)
        self._assert_finite_gmm(self.neg_gmm)
        self.params_vectorized["neg"] = self._vectorize_params(self.neg_gmm)

    def predict(self, X):
        return _diag_gmm_logprob(X, *self.params_vectorized["pos"]) \
             - _diag_gmm_logprob(X, *self.params_vectorized["neg"])

    @staticmethod
    def _vectorize_params(gmm):
        """A fitted pomegranate GMM's params as stacked tensors (log_priors (K,), means (K,d), covs
        (K,d)) for the batched predict, instead of pomegranate's per-component Python loop."""
        return (torch.log(gmm.priors).detach(),
                torch.stack([d.means for d in gmm.distributions]).detach(),
                torch.stack([d.covs for d in gmm.distributions]).detach())

    @staticmethod
    def _assert_finite_gmm(gmm):
        # pomegranate can finish fit() with NaN params (an empty-cluster 0/0 in from_summaries doesn't
        # raise). Such a GMM is a landmine: _initialized is True but log_probability crashes later. Fail
        # loudly at fit time so the run collapses here rather than saving/gating through a broken GMM.
        bad = not torch.isfinite(gmm.priors).all() or any(not torch.isfinite(d.covs).all() or not torch.isfinite(d.means).all() for d in gmm.distributions)
        if bad:
            raise ValueError("GMM fit produced non-finite params (NaN/inf) -- likely an empty-cluster collapse.")

    def get_performance_metrics(self, X_pos, X_neg):
        from sklearn.metrics import confusion_matrix, classification_report
        X = torch.concat([X_pos, X_neg], dim=0)
        y_true = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])], axis=0).astype(int)
        scores = self.predict(X)
        y_pred = (scores > 0).to(torch.int32)
        confusion = confusion_matrix(y_true, y_pred.cpu().numpy())
        report = classification_report(y_true, y_pred.cpu().numpy(), output_dict=True)
        return confusion, report

    # --- (de)serialization ---

    def state_dict(self) -> dict:
        return {
            "pos_kwargs": self.pos_kwargs,
            "neg_kwargs": self.neg_kwargs,
            "pos": gmm_state(self.pos_gmm),
            "neg": gmm_state(self.neg_gmm),
        }

    def load_state_dict(self, state: dict) -> None:
        self.pos_kwargs = dict(state["pos_kwargs"])
        self.neg_kwargs = dict(state["neg_kwargs"])
        if state["pos"] is not None:
            self.load_gmm_state("pos", state["pos"])
        if state["neg"] is not None:
            self.load_gmm_state("neg", state["neg"])

    def load_gmm_state(self, which: str, state: dict) -> None:
        """Restore one GMM (pos/neg) from saved params: the pomegranate object (so it can be refit or
        re-saved) and the vectorized params predict actually uses."""
        gmm = getattr(self, f"{which}_gmm")
        with torch.no_grad():
            gmm.priors = torch.nn.Parameter(state["priors"].to(self.device), requires_grad=False)
            gmm._log_priors = torch.log(gmm.priors)
            for d, mean, cov in zip(gmm.distributions, state["means"], state["covs"]):
                d.means = torch.nn.Parameter(mean.to(self.device), requires_grad=False)
                d.covs = torch.nn.Parameter(cov.to(self.device), requires_grad=False)
                d.d = mean.shape[-1]
                d._initialized = True
                d._reset_cache()
            gmm.d = state["means"][0].shape[-1]
            gmm._initialized = True
        self.params_vectorized[which] = self._vectorize_params(gmm)


def _diag_gmm_logprob(X, log_priors, means, covs):
    """Log-likelihood of X under a diagonal-covariance GMM, all components in one batched op.
    X: (m, d); means/covs: (K, d); log_priors: (K,). Returns (m,)."""
    # per-component log N(x): -0.5 * sum_d [ (x-mu)^2/var + log(2*pi*var) ]
    diff = X.unsqueeze(1) - means.unsqueeze(0)                    # (m, K, d)
    log_comp = -0.5 * ((diff * diff / covs).sum(-1) + torch.log(2 * np.pi * covs).sum(-1))  # (m, K)
    return torch.logsumexp(log_comp + log_priors, dim=1)         # (m,)


# --- GMM (de)serialization ---
# Pull fitted params out as plain tensors rather than pickling pomegranate objects, which would tie
# every checkpoint to a library version.


def gmm_state(gmm) -> Optional[dict]:
    if not gmm._initialized:  # never fitted (e.g. saved before the first fit_cl_memory)
        return None
    return {
        "priors": gmm.priors.detach().cpu(),
        "means": [d.means.detach().cpu() for d in gmm.distributions],
        "covs": [d.covs.detach().cpu() for d in gmm.distributions],
    }


# --- Caching a set of fitted GMMs (pos or neg) ---
# Stores the GMM params, keyed by module name -- the analog of uob's per-layer cl_rad calibration file. 
# Used internally by train_continual.py (e.g. neg and pos calibs are computed at different times, or sometimes on calib is already computed (mostly relevant for neg))
# Unlike save/load_cl_state it does not touch adapters.

def save_gmms(model, path: str, which: str = "neg") -> None:
    """Save one GMM (pos or neg) per adapter, per owner module. With multiphase learning a module holds
    one classifier per phase, so "gmms" is a list indexed by adapter -- saving only cl_gmm[0] would drop
    every phase after the first."""
    from modeling_qwen2 import ConditionalLoRALinear
    base = model.module if hasattr(model, "module") else model
    gmms = {}
    for name, m in base.named_modules():
        if isinstance(m, ConditionalLoRALinear) and m.is_cl_memory_owner:
            gmms[name] = {
                "is_disabled": list(m.is_disabled),  # per-phase: one entry per adapter
                # Module-level and permanent, unlike is_disabled: records that this module's negative fit
                # failed, so a phase added after a reload is disabled on arrival (see add_lora).
                "neg_fit_failed": getattr(m, "_neg_fit_failed", False),
                "gmms": [gmm_state(getattr(clf, f"{which}_gmm")) for clf in m.cl_gmm],
            }
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(gmms, path)


def load_gmms(model, path: str, which: str = "neg") -> None:
    from modeling_qwen2 import ConditionalLoRALinear
    base = model.module if hasattr(model, "module") else model
    gmms = torch.load(path, map_location="cpu", weights_only=False)
    for name, m in base.named_modules():
        if not (isinstance(m, ConditionalLoRALinear) and m.is_cl_memory_owner):
            continue
        saved = gmms[name]
        # Files written before multiphase support hold a single GMM and a single is_disabled bool per module (those runs had one adapter), so both are read as one-element lists.
        saved_gmms = saved["gmms"] if "gmms" in saved else [saved["gmm"]]
        # Permanent module-level verdict: a phase added after this load is disabled on arrival rather than
        # appended enabled and trainable to a module that can never gate. Absent in pre-multiphase files.
        m._neg_fit_failed = saved.get("neg_fit_failed", False)
        saved_disabled = saved["is_disabled"]
        if not isinstance(saved_disabled, list):
            saved_disabled = [bool(saved_disabled)]
        # A length mismatch means this file does not belong to this model (different number of phases), so
        # fail loudly rather than restore some phases and leave the rest at whatever they happened to be.
        assert len(saved_disabled) == len(m.is_disabled), \
            f"{name}: saved is_disabled has {len(saved_disabled)} phases, model has {len(m.is_disabled)}"
        # In place, so the siblings aliasing the owner's list keep pointing at it. An adapter whose fit
        # failed at save time stays disabled (that phase's LoRA is skipped in the forward).
        m.is_disabled[:] = saved_disabled
        for clf, gmm in zip(m.cl_gmm, saved_gmms):
            if gmm is not None:
                clf.load_gmm_state(which, gmm)


def is_in_gmm(clf: GMMClassifierGPU, new_samples: torch.Tensor, chunk_m: int = 131072) -> torch.Tensor:
    """Membership via the positive/negative likelihood ratio. Mirrors ``is_in_union_of_balls``'
    contract: (m, d) in, (m, 1) int32 out."""
    m = new_samples.shape[0]
    if not clf.is_fitted:  # not fitted yet (e.g. task 0's zero-shot eval, before the first pos fit) -> nothing in memory, like an empty union-of-balls buffer
        return torch.zeros(m, 1, dtype=torch.int32, device=new_samples.device)
    out = torch.empty(m, dtype=torch.int32, device=new_samples.device)
    for s in range(0, m, chunk_m):  # pomegranate needs fp32; upcast one tile at a time
        out[s:s + chunk_m] = (clf.predict(new_samples[s:s + chunk_m].float()) > 0).to(torch.int32)
    return out.unsqueeze(dim=1)


# ---------------------------------------------------------------------------
# Fitting: layer-sequential recomputation
# ---------------------------------------------------------------------------
# Qwen2Model.forward, given `full_dataset_input`, walks the layers with the minibatches on the
# inside and calls fit_layer_gmms() after each layer. So a layer's activations are all collected,
# fitted and freed before the next layer runs, and only one layer's worth is ever live.


def fit_layer_gmms(decoder_layer, which: str = "pos", skip_on_fail: bool = False) -> None:
    """Fit the GMMs of every cl_memory owner in this layer, then release their activations.

    The activations were already captured by the accumulation branch in
    ``ConditionalLoRALinear.forward``, which projects through jl_P and drops pad tokens -- exactly
    the points we want to fit -- so nothing needs to be recomputed here.

    ``skip_on_fail``: if a module's fit raises an error, disable that module's LoRA and continue, instead of killing the run.
    A disabled LoRA is fully inert in the forward, so its unfitted GMM is never gated through.
    """
    from modeling_qwen2 import ConditionalLoRALinear

    owners = [(n, m) for n, m in decoder_layer.named_modules()
              if isinstance(m, ConditionalLoRALinear) and m.is_cl_memory_owner
              and not m.is_disabled[m.accum_lora_idx] and m._pending_x]
    layer_idx = getattr(decoder_layer.self_attn, "layer_idx", "?")
    for name, module in tqdm(owners, desc=f"fit {which} gmms L{layer_idx}", leave=False):
        X = torch.vstack(module._pending_x).float()  # pomegranate needs fp32
        clf = module.cl_gmm[module.accum_lora_idx]
        try:
            clf.fit_pos(X) if which == "pos" else clf.fit_neg(X)
        except Exception as e:
            if not skip_on_fail:
                raise
            logger.warning(f"[fit_cl_memory] L{layer_idx} {name}: {which} fit failed ({e}); disabling this phase's LoRA")
            # Disable ONLY this phase's adapter, so the forward skips it (no gating through an unfitted GMM)
            module.is_disabled[module.accum_lora_idx] = True
            if which == "neg":
                # The negative is fitted once per run (phase 0) and never refitted, so without it this module
                # has no density ratio and NO future phase can gate either -- unlike a failed positive, which
                # the next phase refits. Recorded explicitly so add_lora can disable this module's later
                # phases on arrival; is_disabled alone cannot say which of the two fits failed.
                module._neg_fit_failed = True
        module._pending_x = []  # free before the next layer runs
        del X
    torch.cuda.empty_cache()


def shared_prefix_ids(tokenizer) -> torch.Tensor:
    """Token ids of the chat template's shared prefix: everything it emits BEFORE the prompt.

    Empty when the tokenizer has no chat template (a base, non-instruct model): there is then no shared
    prefix to hold out, and drop_shared_prefix_mask matches nothing.
    """
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        return torch.empty(0, dtype=torch.long)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)
    header = "<|im_start|>user\n"
    prefix = rendered[:rendered.index(header) + len(header)] if header in rendered else rendered
    return torch.tensor(tokenizer(prefix, add_special_tokens=False)["input_ids"])


def drop_shared_prefix_mask(input_ids, attention_mask, prefix_ids) -> torch.Tensor:
    """[B, T] bool marking the tokens to KEEP: every real token except the shared template prefix.

    The prefix is MATCHED, not assumed: a row's leading real tokens are only dropped if they equal
    prefix_ids. An untemplated corpus, or a row whose front was cut off by left truncation, does not match
    and is kept whole -- so real content is never mistaken for scaffolding.

    There are 5 template tokens after the prompt which we dont drop - this is because (1) it requires more care to find them and (2)
    they are not "shared" anymore in the sense that the causal model representations may already differ at this point.
    """
    keep = attention_mask.bool().clone()
    n = prefix_ids.shape[0]
    if n == 0:  # no chat template -> nothing to hold out, every real token counts
        return keep
    for b in range(input_ids.shape[0]):
        head = torch.nonzero(keep[b], as_tuple=True)[0][:n]  # left padding: the first REAL tokens
        if head.shape[0] == n and torch.equal(input_ids[b, head], prefix_ids):
            keep[b, head] = False
    return keep


@torch.no_grad()
def gmm_eval(model, pos_dl, neg_dl, tokenizer, config, pt_tasks_dls=None) -> dict:
    """Hit rate of each module's gating on held-out positive and negative data.

    Runs plain forwards with hit-rate tracking on and reads the gating decisions the forward already
    makes -- no activations are collected. The positive hit rate is the fraction of in-task tokens
    the LoRA fires on (want high); the negative rates are the fraction of out-of-task tokens it wrongly
    fires on (want low): "neg" on a generic corpus, and (when pt_tasks_dls is given) one "pt_<benchmark>"
    rate per pretraining-task benchmark, on the model's own lm-eval generations -- a harder, more
    relevant out-of-task distribution.

    The chat template's shared prefix is excluded from every measured hit rate. An identical prefix for a causal
    model is not distinguishable anyway, and it just inflates the hit rates with useless information.

    pt_tasks_dls: {benchmark_name: dataloader} or None.
    Returns {module_name: {"pos": rate, "neg": rate[, "pt_<benchmark>": rate, ...]}}.
    """
    from modeling_qwen2 import ConditionalLoRALinear

    base = model.module if hasattr(model, "module") else model
    device = next(base.parameters()).device
    was_training = base.training
    base.eval()

    owners = [(n, m) for n, m in base.named_modules()
              if isinstance(m, ConditionalLoRALinear) and m.is_cl_memory_owner
              and any(not d for d in m.is_disabled)]  # at least one phase still gating

    def hit_rates(dataloader, desc):
        for _, module in owners:
            module.hit_count = module.token_count = 0
            module.seq_hit_count = module.seq_count = 0
            module.is_hit_rate_tracking = True
        for cur_mb in tqdm(dataloader, desc=f"gmm_eval {desc}", leave=False):
            enc = tokenizer(cur_mb["text"], return_tensors="pt", max_length=config.seq_len_train,
                            padding="longest", padding_side="left", truncation=True,
                            add_special_tokens=False).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                base(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        rates = {n: (m.hit_count / max(m.token_count, 1), m.seq_hit_count / max(m.seq_count, 1)) for n, m in owners}
        for _, module in owners:
            module.is_hit_rate_tracking = False
        return rates

    pos_rates, neg_rates = hit_rates(pos_dl, "pos"), hit_rates(neg_dl, "neg")
    # "<key>" is the per-token rate; "<key>_seq" the fraction of sequences with >=1 hit.
    out = {name: {"pos": pos_rates[name][0], "pos_seq": pos_rates[name][1],
                  "neg": neg_rates[name][0], "neg_seq": neg_rates[name][1]} for name in pos_rates}
    for benchmark, dl in (pt_tasks_dls or {}).items():
        bench_rates = hit_rates(dl, f"pt_{benchmark}")
        for name in out:
            out[name][f"pt_{benchmark}"] = bench_rates[name][0]
            out[name][f"pt_{benchmark}_seq"] = bench_rates[name][1]

    if was_training:
        base.train()
    return out


def report_gmm_eval(scores: dict, logger=None, wandb_log=None) -> None:
    """Log the gmm_eval hit rates: per-module-type and per-layer summaries to the log, plus the means to wandb."""
    if not scores:
        return
    sample = next(iter(scores.values()))
    keys = ["pos", "pos_seq", "neg", "neg_seq"] + sorted(k for k in sample if k.startswith("pt_"))  # per-benchmark pt_<name>[_seq] keys present only when pt_tasks_dls was given
    means = {k: np.mean([r[k] for r in scores.values()]) for k in keys}
    by_type, by_layer = {}, {}
    for name, rates in scores.items():
        by_type.setdefault(name.rsplit(".", 1)[-1], []).append(rates)
        parts = name.split(".")  # ...layers.<i>.<...>
        if "layers" in parts:
            by_layer.setdefault(int(parts[parts.index("layers") + 1]), []).append(rates)

    if logger is not None:
        summary = "  ".join(f"{t}: " + "/".join(f"{np.mean([r[k] for r in v]):.4f}" for k in keys)
                            for t, v in sorted(by_type.items()))
        overall = "/".join(f"{means[k]:.3f}" for k in keys)
        logger.info(f"[gmm_eval] hit rate {'/'.join(keys)}: {overall} | per module type: {summary}")
        layer_summary = "  ".join(f"L{l}: " + "/".join(f"{np.mean([r[k] for r in v]):.4f}" for k in keys)
                                  for l, v in sorted(by_layer.items()))
        logger.info(f"[gmm_eval] hit rate {'/'.join(keys)} per layer: {layer_summary}")
    if wandb_log is not None:
        log = {f"gmm_eval/hit_rate_{k}": means[k] for k in keys}
        for t, v in by_type.items():
            for k in keys:
                log[f"gmm_eval/{t}_{k}"] = np.mean([r[k] for r in v])
        for l, v in by_layer.items():
            for k in keys:
                log[f"gmm_eval_per_layer/L{l}_{k}"] = np.mean([r[k] for r in v])
        wandb_log(log)


@torch.no_grad()
def fit_cl_memory(model, dataloader, tokenizer, config, which: str = "pos") -> float:
    """Refit every wrapped module's GMM on a subsample of the current dataset.

    Pulls batches until ``config.gmm_fit_num_tokens`` real (non-pad) tokens have been collected,
    then hands them to forward() as ``full_dataset_input`` so the layers are swept and fitted one at
    a time. ``which`` selects which of each classifier's GMMs the activations train.

    Returns the fraction of owner modules still active (not disabled) after fitting.
    """
    from modeling_qwen2 import ConditionalLoRALinear, set_cl_accum_state, set_cl_accum_off

    base = model.module if hasattr(model, "module") else model
    model_backbone = base.model  # Qwen2Model
    device = next(model_backbone.parameters()).device
    was_training = base.training
    base.train()

    # Collect the subsample, counting only real tokens: pad tokens are never fitted on, so they must
    # not eat the budget. Tokenizer settings match the training loop so the activations we fit on are
    # the ones the model actually sees. Batches stay separate (each padded to its own width) --
    # stacking them would pad everything to the widest sequence in the whole subsample.
    full_dataset_input, num_tokens = [], 0
    while num_tokens < config.gmm_fit_num_tokens:
        cur_mb, epoch_ended = next(dataloader)
        enc = tokenizer(cur_mb["text"], return_tensors="pt", max_length=config.seq_len_train,
                        padding="longest", padding_side="left", truncation=True,
                        add_special_tokens=False).to(device)
        full_dataset_input.append({"input_ids": enc.input_ids, "attention_mask": enc.attention_mask})
        num_tokens += int(enc.attention_mask.sum())
        if epoch_ended:  # don't spill into the next epoch just to top up the sample
            break

    logger.info(f"[fit_cl_memory] {which}: collected {num_tokens} tokens over {len(full_dataset_input)} batches "
                f"(budget {config.gmm_fit_num_tokens}{'; SPLIT EXHAUSTED FIRST' if num_tokens < config.gmm_fit_num_tokens else ''})")

    set_cl_accum_state(base, True)  # forward() buffers its inputs into _pending_x for us
    for module in base.modules():   # start clean: no leftovers from training
        if isinstance(module, ConditionalLoRALinear):
            module._pending_x = []
    model_backbone.gmm_fit_which = which
    model_backbone.gmm_skip_on_fail = getattr(config, "gmm_skip_on_fail", False)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        model_backbone(full_dataset_input=full_dataset_input)

    # Report which modules stayed active vs. got disabled by skip_on_fail (owners only; siblings share the owner's decision).
    owners = [(n, m) for n, m in base.named_modules()
              if isinstance(m, ConditionalLoRALinear) and m.is_cl_memory_owner]
    # Active/disabled is reported for the phase just fitted (earlier phases were settled by earlier runs).
    active = [n for n, m in owners if not m.is_disabled[m.accum_lora_idx]]
    active_ratio = len(active) / max(len(owners), 1)
    logger.info(f"[fit_cl_memory] {which}: {len(active)}/{len(owners)} active owner modules "
                f"({100 * active_ratio:.1f}%)")
    disabled = [n for n, m in owners if m.is_disabled[m.accum_lora_idx]]
    if disabled:
        logger.info(f"[fit_cl_memory] {which}: disabled modules: {disabled}")

    set_cl_accum_off(base)

    del full_dataset_input
    for module in base.modules():
        if isinstance(module, ConditionalLoRALinear):
            module._pending_x = []
    torch.cuda.empty_cache()

    base.train() if was_training else base.eval()

    return active_ratio
