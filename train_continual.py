import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, get_wsd_schedule
from modeling_qwen2 import Qwen2ForCausalLM, apply_conditional_lora_wrapper, add_lora, set_shared_prefix_ids, update_cl_memory, flush_cl_memory, set_cl_accum_on, set_cl_accum_off, set_cl_accum_state, get_cl_accum_state, report_cl_memory, report_compaction, cl_memory_breakdown, save_cl_state, load_cl_state
from cl_memory import fit_cl_memory, gmm_eval, report_gmm_eval, save_gmms, load_gmms
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader, DistributedSampler
import time
from loguru import logger
import wandb
from safetensors.torch import save_file
import os
from wandb_tools import wandb_init, wandb_log, wandb_commit
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import copy
import math
import json
import sacrebleu

from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

from ddp import wrap_model_ddp, dist_group, all_reduce, all_gather_list, barrier
from utils import get_grad_norms, get_param_count, get_mlp_frobenius_norms
from utils import log_finetune_eval_summary, log_continual_learning_eval_summary

from torch.distributed.optim import ZeroRedundancyOptimizer

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
use_ddp = world_size > 1

PRETRAINING_DATASETS = ['fineweb_edu', 'lsl_negatives']

def is_cybersecurity_ift(dataset_name):
    return dataset_name == 'cybersecurity_ift'

def is_seceval(dataset_name):
    return dataset_name == 'seceval'

def is_sft(dataset_name):
    return dataset_name in ('chemdata', 'igbo_translation', 'cybersecurity_ift')

def is_chembench(dataset_name):
    return dataset_name == 'chembench'

# The translation benchmark paired with the igbo_translation SFT dataset (see DATASET_BENCHMARK). Scored by
# corpus chrF over generated translations (see generative_eval), on the dataset's own held-out test split.
def is_igbo_translation_benchmark(dataset_name):
    return dataset_name == 'igbo_translation_benchmark_version'

# Some training datasets carry an associated held-out benchmark that is evaluated alongside the dataset's own eval (e.g. chemdata's SFT answer-PPL is reported together with ChemBench4K accuracy).
DATASET_BENCHMARK = {'chemdata': 'chembench', 'igbo_translation': 'igbo_translation_benchmark_version', 'cybersecurity_ift': 'seceval'}

# Collate datasets whose examples are prompt+single-answer QA: mask the prompt so we only train/score the answer token (see the masking branches in ppl_eval / the train loop).
QA_DATASET_TAGS = ['chembench']

# Free-form SFT datasets: mask the prompt so we train/score over the full (multi-token) answer span.
SFT_DATASET_TAGS = ['sft']

# All QA datasets columns are standardized for PreppedQACollate and concatenated for IID training (see load_dataset_splits).
PREPPED_QA_COLUMNS = ['prompt', 'cand_labels', 'answer']

# Free-form SFT datasets are standardized to {prompt, answer} for SFTCollate (no cand_labels).
PREPPED_SFT_COLUMNS = ['prompt', 'answer']

# Per-dataset test split size for train_test_split: a float is a fraction of the data,
# an int is an absolute number of eval samples. Datasets with a built-in test split are omitted.
DATASET_TEST_SIZE = {
    # Trendyol ships its own train/validation/test splits, so nothing is carved out of train; this only caps how much of the 2660-row test split the answer-PPL eval uses.
    'cybersecurity_ift': 1000,
    'seceval': 500,  # held-out MCQ eval count from SecEval's single 2189-row split
    'chembench': 300, #1000,  # held-out eval count from the 4009 ChemBench MCQs
    'chemdata': 1000,  # absolute eval count: ChemData700K is large, so a fraction would be an enormous eval set
    # igbo ships its own train/test split, so nothing is carved out of train; these only cap how much of
    # the 1192-row test split each eval uses, in line with the other datasets.
    'igbo_translation_benchmark_version': None,  # kept small: chrF over generated translations is far slower than a forward pass
    'fineweb_edu': 400,
    'lsl_negatives': 400,
}

CHEMDATA_MAX_TRAIN = 20000 #100000 # None = use all samples.
BLEND_CHEMBENCH_TRAIN = True  # mix the ChemBench MCQ train pool into chemdata training (forward learning on the benchmark); held-out ChemBench test stays eval-only.

# Per-benchmark generation budget for generative_eval
DATASET_MAX_GEN_LENGTH = {
    'igbo_translation_benchmark_version': 64,
    'seceval': 8,  # only the option letters are parsed off the front of the completion
}

# Tommy0201/igbo_to_english_split translation. Scored by corpus chrF over generated translations.
# Direction matters for what the model actually learns: the prompt span is masked out (see SFTCollate), so
# only the target side is ever predicted. Igbo->English puts gradient on English tokens given Igbo context;
# English->Igbo is the direction that trains on the Igbo tokens themselves.
TRANSLATE_IGBO_TO_ENG = False  # True: Igbo -> English. False: English -> Igbo (predict the Igbo tokens).
IGBO_TRANSLATION_INSTRUCTION = ("Translate the following Igbo sentence into English." if TRANSLATE_IGBO_TO_ENG
                                else "Translate the following English sentence into Igbo.")
IGBO_TRANSLATION_MAX_TRAIN = 60000  # cap on the translation SFT train pool. None = use all ~117k train rows.

# How many (prompt, generation, gold) triples generative_eval logs from its first batch. The generated text is
# the only way to tell a wrong answer apart from a right answer in the wrong format (which scores 0 either way).
GENERATIVE_EVAL_NUM_SAMPLES = 5

def load_dataset_splits(dataset_name, seed):
    """Load and return train/test splits for a given dataset name."""
    if dataset_name == 'fineweb_edu':
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT")
    elif dataset_name == 'lsl_negatives':
        dataset = load_dataset("assafbk/lsl-negatives-fineweb-edu", split="train")
    elif dataset_name == 'chemdata':  # free-form SFT (AI4Chem/ChemData700K): instruction+input -> output, trained/scored over the full answer
        raw_dataset = load_dataset("AI4Chem/ChemData700K", split="train")
        raw_dataset = raw_dataset.map(_chemdata_to_prepped, remove_columns=raw_dataset.column_names)
        if CHEMDATA_MAX_TRAIN is not None:  # cap before splitting so the eval set is carved from the same (shuffled) subset
            raw_dataset = raw_dataset.shuffle(seed=seed).select(range(min(CHEMDATA_MAX_TRAIN + DATASET_TEST_SIZE['chemdata'], len(raw_dataset))))
        splits = raw_dataset.train_test_split(test_size=DATASET_TEST_SIZE['chemdata'], seed=seed, shuffle=True)
        if BLEND_CHEMBENCH_TRAIN:  # mix the ChemBench MCQ train pool (rendered as SFT) into training so the model also learns the benchmark task; its held-out test set stays eval-only
            splits["train"] = concatenate_datasets([splits["train"], load_dataset_splits('chembench', seed)["train"]]).shuffle(seed=seed)
    elif dataset_name == 'chembench':  # AI4Chem/ChemBench4K MCQ. Split the 4009 into a held-out test (argmax eval) and a train pool (rendered as SFT, blended into chemdata training). The test set is NEVER trained on.
        raw_dataset = load_dataset("AI4Chem/ChemBench4K", split="test")
        raw_splits = raw_dataset.train_test_split(test_size=DATASET_TEST_SIZE['chembench'], seed=seed, shuffle=True)
        fewshot_prefix = _chembench_fewshot_prefix(CHEMBENCH_NUM_FEWSHOT, seed)  # eval-time few-shot scaffold; same demos for every test row, drawn from the validation split
        test = raw_splits["test"].map(lambda row: _chembench_to_prepped(row, fewshot_prefix), remove_columns=raw_dataset.column_names)  # QA schema for PreppedQACollate argmax eval
        train = raw_splits["train"].map(_chembench_to_sft, remove_columns=raw_dataset.column_names)  # SFT schema (0-shot, answer = letter) for blending into training
        splits = DatasetDict({"train": train, "test": test})
    elif dataset_name == 'igbo_translation':  # Tommy0201/igbo_to_english_split as free-form SFT: Igbo sentence -> English translation, trained/scored over the translation. Its paired BLEU benchmark is igbo_translation_benchmark_version.
        raw_dataset = load_dataset("Tommy0201/igbo_to_english_split")
        train = raw_dataset["train"]
        if IGBO_TRANSLATION_MAX_TRAIN is not None:  # shuffle first so the cap is a random subset, not the head of the split
            train = train.shuffle(seed=seed, keep_in_memory=True).select(range(min(IGBO_TRANSLATION_MAX_TRAIN, len(train))), keep_in_memory=True)
        splits = DatasetDict({  # the dataset ships its own train/test split: train for SFT, test for the answer-PPL eval
            "train": train.map(_igbo_translation_to_prepped, remove_columns=train.column_names,keep_in_memory=True),
            "test": raw_dataset["test"].map(_igbo_translation_to_prepped, remove_columns=raw_dataset["test"].column_names, keep_in_memory=True),
        })
    elif is_igbo_translation_benchmark(dataset_name):  # translation benchmark: the same held-out test rows, scored by generating the translation and computing corpus chrF (see generative_eval). Eval-only, never trained on.
        raw_dataset = load_dataset("Tommy0201/igbo_to_english_split", split="test")
        test = raw_dataset.map(_igbo_translation_to_prepped, remove_columns=raw_dataset.column_names, keep_in_memory=True)
        # Cap the generative eval: decoding a translation per row is far slower than a forward pass, and this runs once per CL phase. Shuffle first so the subset is representative, not the head of the split.
        cap = DATASET_TEST_SIZE.get(dataset_name)
        if cap is not None:
            test = test.shuffle(seed=seed, keep_in_memory=True).select(range(min(cap, len(test))), keep_in_memory=True)
        splits = DatasetDict({"train": test.select(range(0), keep_in_memory=True), "test": test})  # empty train: benchmark is eval-only
    elif is_cybersecurity_ift(dataset_name):  # Trendyol cybersecurity IFT as free-form SFT: security question -> full technical answer, trained/scored over the answer. Its paired MCQ benchmark is seceval.
        raw_dataset = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train")  # ships one 53201-row split, so the eval set is carved out below
        raw_dataset = raw_dataset.map(_cybersecurity_ift_to_prepped, remove_columns=raw_dataset.column_names, keep_in_memory=True)
        if CYBERSECURITY_IFT_MAX_TRAIN is not None:  # cap before splitting so the eval set is carved from the same (shuffled) subset
            raw_dataset = raw_dataset.shuffle(seed=seed, keep_in_memory=True).select(range(min(CYBERSECURITY_IFT_MAX_TRAIN + DATASET_TEST_SIZE['cybersecurity_ift'], len(raw_dataset))), keep_in_memory=True)
        splits = raw_dataset.train_test_split(test_size=DATASET_TEST_SIZE['cybersecurity_ift'], seed=seed, shuffle=True)
        if BLEND_SECEVAL_TRAIN:  # same, for the paired SecEval benchmark: without it the model only ever produces long-form prose and its option-letter behaviour degrades even as answer-PPL falls
            splits["train"] = concatenate_datasets([splits["train"], load_dataset_splits('seceval', seed)["train"]]).shuffle(seed=seed)
    elif is_seceval(dataset_name):  # security-knowledge MCQ benchmark, scored by generating the option letters and computing set F1 (see generative_eval). Ships a single 2189-row split, so it is divided here into a held-out test set and a train pool (rendered as SFT, blended into cybersecurity_ift training). The test set is NEVER trained on.
        raw_dataset = load_dataset("XuanwuAI/SecEval", split="train")
        raw_dataset = raw_dataset.filter(_seceval_is_scorable, keep_in_memory=True)  # drop rows without 4 options / an A-D gold
        raw_splits = raw_dataset.train_test_split(test_size=DATASET_TEST_SIZE['seceval'], seed=seed, shuffle=True)  # shuffled, so both sides span all 9 domains
        splits = DatasetDict({
            "train": raw_splits["train"].map(_seceval_to_prepped, remove_columns=raw_splits["train"].column_names, keep_in_memory=True),  # SFT schema: blended into cybersecurity_ift training
            "test": raw_splits["test"].map(_seceval_to_prepped, remove_columns=raw_splits["test"].column_names, keep_in_memory=True),  # scored by generation + set F1
        })
    else:
        raise ValueError(f'{dataset_name} dataset not supported')

    # Standardize every QA dataset to PREPPED_QA_COLUMNS (chembench is already rendered
    # per-split above — their train is SFT schema and test is QA schema, so they are exempt from this
    # whole-DatasetDict column strip).
    if any(tag in dataset_name for tag in QA_DATASET_TAGS) and not is_chembench(dataset_name):
        splits = splits.remove_columns([c for c in splits["train"].column_names if c not in PREPPED_QA_COLUMNS])
    if is_sft(dataset_name):
        splits = splits.remove_columns([c for c in splits["train"].column_names if c not in PREPPED_SFT_COLUMNS])

    if dataset_name in PRETRAINING_DATASETS:
        splits = dataset.train_test_split(test_size=DATASET_TEST_SIZE[dataset_name], seed=seed, shuffle=True)

    return splits

def multi_epoch_loader(dl):
    """Yields (batch, is_epoch_ended) forever, cycling through epochs.
    There is a buffer for the next batch - this way we always return samples even when the epoch is finished.
    """
    while True:
        next_batch = None
        for batch in dl:
            if next_batch is None:
                next_batch = batch
                continue
                
            cur_batch = next_batch
            next_batch = batch
            
            yield cur_batch, False

        yield next_batch, True # Signal epoch boundary — caller should stop accumulating and step

def build_gmm_kwargs(config):
    """pos/neg GMM settings for the gmm cl_memory backend (None when the backend is 'uob').
    The two GMMs differ only in component count: the positive models one task, the negative a
    broad corpus. Distinct seeds keep their inits independent."""
    if config.cl_memory_type != 'gmm':
        return None
    return dict(
        pos_kwargs=dict(n_components=config.gmm_n_pos_comps, covariance_type=config.gmm_covariance_type, random_state=config.gmm_seed),
        neg_kwargs=dict(n_components=config.gmm_n_neg_comps, covariance_type=config.gmm_covariance_type, random_state=config.gmm_seed + 1),
    )

def get_collate_fn_per_dataset(dataset_name, tokenizer):
    if is_chembench(dataset_name):
        return PreppedQACollate(tokenizer, 'chembench', answer_prefix=CHEMBENCH_ANSWER_PREFIX)
    if is_seceval(dataset_name):
        return GenerativeCollate(tokenizer, 'seceval')
    if is_igbo_translation_benchmark(dataset_name):
        return GenerativeCollate(tokenizer, 'igbo_translation_benchmark_version')
    if is_sft(dataset_name):
        return SFTCollate(tokenizer, SFT_DATASET_TAGS[0])
    return None


def make_dl(dataset, batch_size, is_train, collate_fn):
    """Build one DataLoader with the shared worker config (MNIST_train.py style).
    Under DDP, a DistributedSampler handles both per-rank splitting and shuffling (so the
    dataset is NOT pre-sharded and shuffle stays False). multiprocessing_context='fork'
    requires this to be called BEFORE CUDA init."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=is_train) if use_ddp else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None and is_train), drop_last=is_train,
        prefetch_factor=2, num_workers=4, pin_memory=True, persistent_workers=True,
        multiprocessing_context='fork', collate_fn=collate_fn,
    )


def make_text_dl(texts, batch_size):
    """Batch a plain list of strings into the {"text": [...]} minibatches the eval loaders yield.
    Used for the pt_tasks eval set (captured lm-eval generations). None if there are no texts."""
    if not texts:
        return None
    return [{"text": texts[i:i + batch_size]} for i in range(0, len(texts), batch_size)]


def build_dataloaders(config, tokenizer):
    """Pre-build all eval dataloaders BEFORE CUDA is initialized, so the fork-based
    workers spawn from a CUDA-clean process (avoids the fork-after-CUDA hang).

    One (train, test) pair per dataset, gated by config flags. Continual learning
    and transfer learning (finetune) both loop over datasets, so they share the
    same keying; the CL loop additionally reuses other tasks' test loaders for cross-eval.
    Returns {dataset: (multi_epoch_train_dl, test_dl, train_size)}.
    """
    def add(dataset, name=None):
        key = name if name else dataset
        if key in dataloaders:
            return
        splits = load_dataset_splits(dataset, config.seed)
        collate = get_collate_fn_per_dataset(dataset, tokenizer)
        train_dl = make_dl(splits["train"], config.local_batch_size, is_train=True, collate_fn=collate)
        test_dl  = make_dl(splits["test"], config.eval_batch_size, is_train=False, collate_fn=collate)
        train_size = len(train_dl.sampler) if use_ddp else len(splits["train"]) # Per-rank train size for step math: DistributedSampler reports the per-rank count under DDP.
        dataloaders[key] = (multi_epoch_loader(train_dl), test_dl, train_size)
        if config.cl_memory_type == 'gmm' and config.use_conditional_lora and config.activate_cl_method:
            pos_fit_dl = make_dl(splits["train"], config.local_batch_size, is_train=True, collate_fn=collate)
            dataloaders[f'gmm_pos_{key}'] = (multi_epoch_loader(pos_fit_dl), None, train_size)

    def add_benchmark(dataset):  # Build the held-out benchmark's test loader (eval only, never trained on) for a dataset that has one.
        benchmark = DATASET_BENCHMARK.get(dataset)
        if benchmark is None or benchmark in dataloaders:
            return
        splits = load_dataset_splits(benchmark, config.seed)
        collate = get_collate_fn_per_dataset(benchmark, tokenizer)
        dataloaders[benchmark] = (None, make_dl(splits["test"], config.eval_batch_size, is_train=False, collate_fn=collate), 0)

    dataloaders = {}
    if config.run_cl_eval:
        for dataset in config.continual_learning_datasets:
            add(dataset)
            add_benchmark(dataset)
    if config.cl_memory_type == 'gmm' and config.use_conditional_lora and config.activate_cl_method:
        # Negative corpus for GMM training. Map fineweb_edu rows to SFT schema — labels are ignored, we only need activations
        dataset_tag = 'lsl_negatives' if config.use_prepared_negatives_set else 'fineweb_edu'
        neg_splits = load_dataset_splits(dataset_tag, config.seed)
        neg_train_cap = min(len(neg_splits['train']), 4 * config.gmm_fit_num_tokens // config.seq_len_train) # Cap the train split BEFORE mapping - the overhead of x4 is enough (violated only when the average seq len is config.seq_len_train // 4 which is unlikely)
        neg_splits['train'] = neg_splits['train'].select(range(neg_train_cap))
        neg_splits = neg_splits.map(lambda r: {"prompt": r["text"], "answer": ""}, keep_in_memory=True)  # every rank maps the same split concurrently; a cache file would be written/unlinked by several ranks at once
        neg_collate = SFTCollate(tokenizer, dataset_tag=dataset_tag) # use the default sft collater so we keep the same format (chat_template, etc.)
        neg_train_dl = make_dl(neg_splits['train'], config.local_batch_size, is_train=True, collate_fn=neg_collate)
        neg_test_dl = make_dl(neg_splits['test'], config.eval_batch_size, is_train=False, collate_fn=neg_collate)
        neg_train_size = len(neg_train_dl.sampler) if use_ddp else len(neg_splits['train'])
        dataloaders['gmm_neg'] = (multi_epoch_loader(neg_train_dl), neg_test_dl, neg_train_size)
    return dataloaders

@torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
# @torch.compiler.set_stance("force_eager")
def ppl_eval(model, tokenizer, test_dl, config, max_length=512, log_prefix="Performance", dataset_name=""):
    """currently assumes a SFT instruct dataset (prompt + answer), masks prompt labels"""
    model.eval()
    if config.activate_cl_method:
        cl_accum_state = get_cl_accum_state(model)
        set_cl_accum_off(model)

    total_nll = 0.0
    total_tokens = 0

    ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    pbar = tqdm(total=len(test_dl), desc="Evaluating PPL", leave=False)
    for batch in test_dl:
        enc = tokenizer(batch["text"], return_tensors="pt", padding="longest", padding_side="left", truncation=True, max_length=max_length, add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100  # ignore pads by mask
        seq_len = labels.shape[1]
        for row_i, plen in enumerate(batch["prompt_len"]):  # ignore prompt tokens; score only the answer span
            pad_offset = int((enc["attention_mask"][row_i] == 0).sum().item())
            labels[row_i, :min(pad_offset + plen, seq_len)] = -100
        shift_labels = torch.hstack([labels[:,1:],torch.full([labels.shape[0],1], -100, device=labels.device)])
        num_tokens = (shift_labels != -100).sum().item()
        total_tokens += num_tokens

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            flat_logits = output.logits.view([-1, output.logits.shape[-1]])
            flat_labels = shift_labels.contiguous().view(-1)
            scored = flat_labels != -100
            total_nll += ce_loss(flat_logits[scored].float(), flat_labels[scored]) # Gather only scored positions before .float()

        if local_rank == 0:
            pbar.update(1)

    total_nll = all_reduce(total_nll, normalize=False)
    total_tokens = all_reduce(total_tokens, normalize=False)

    # PPL
    ppl = torch.exp(total_nll / max(1, total_tokens)).cpu().tolist()

    if local_rank == 0:
        logger.info(f"\n[EVAL] {dataset_name} answer PPL: {ppl:.2f}")
        wandb_log({f"{log_prefix}_ppl/{dataset_name}": ppl})

    model.train()
    if config.activate_cl_method:
        set_cl_accum_state(model, cl_accum_state)
    torch.cuda.empty_cache()

    return ppl


CHEMBENCH_CHOICES = ["A", "B", "C", "D"]
CHEMBENCH_INSTRUCTION = "The following is a multiple-choice question about chemistry. Choose the correct answer from A, B, C, or D."
# Primes the assistant turn so the next token is the answer letter. The trailing quote forces a no-space
# continuation, matching the bare-letter tokens task_eval scores (see PreppedQACollate.answer_prefix).
CHEMBENCH_ANSWER_PREFIX = "" #"The correct answer is the letter '"
CHEMBENCH_NUM_FEWSHOT = 0

def _chembench_render_question(row):
    """The instruction + question + rendered A/B/C/D options block (shared by the demos and the test prompt)."""
    options = "\n".join(f"{c}. {row[c]}" for c in CHEMBENCH_CHOICES)
    return f"{CHEMBENCH_INSTRUCTION}\n\n{row['question']}\n\n{options}"


def _chembench_fewshot_prefix(n, seed):
    """Build the n-shot demo from the ChemBench val split (no overlap with the test set). Returns '' if n == 0."""
    if n <= 0:
        return ""
    val = load_dataset("AI4Chem/ChemBench4K", split="validation").shuffle(seed=seed).select(range(n))
    demos = [f"{_chembench_render_question(row)}\n{CHEMBENCH_ANSWER_PREFIX}{row['answer']}'" for row in val]
    return "\n\n".join(demos) + "\n\n"


def _chembench_to_prepped(row, fewshot_prefix=""):
    """Render one raw AI4Chem/ChemBench4K row into the prepped QA schema ({prompt, cand_labels, answer})."""
    prompt = fewshot_prefix + _chembench_render_question(row)
    return {"prompt": prompt, "cand_labels": CHEMBENCH_CHOICES, "answer": row["answer"]}


def _chembench_to_sft(row):
    """Render one raw AI4Chem/ChemBench4K row into the SFT schema ({prompt, answer}) so it can be blended into
    chemdata training under a single SFTCollate. The answer carries the answer-prefix + gold letter, so the
    trained text reproduces exactly the prefix->letter transition the argmax eval scores (0-shot: no demos in
    training; few-shot is an eval-time scaffold). The answer span (prefix + one letter) is what we train on."""
    return {"prompt": _chembench_render_question(row), "answer": CHEMBENCH_ANSWER_PREFIX + row["answer"]}


# Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset as free-form SFT: a security question prompts the
# model and it is trained/scored over the full technical answer (~700 tokens), across 200+ security subdomains.
# Rows are {system, user, assistant}; `system` is a fixed role/ethics preamble shared by every row, so it is
# dropped rather than prepended (a constant prefix carries no gradient and would sit in cl_memory's shared
# prefix span). Its paired MCQ benchmark is seceval.
CYBERSECURITY_IFT_MAX_TRAIN = None  # cap on the SFT train pool. None = use all ~53.2k rows (minus the carved-out eval set).
BLEND_SECEVAL_TRAIN = True  # mix the SecEval MCQ train pool into training (forward learning on the benchmark); its held-out test set stays eval-only
# Only one MCQ pool may be blended: DATASET_BENCHMARK pairs cybersecurity_ift with a single benchmark, so
# blending the other one would train on a task nothing reports, and mix two option-letter formats.
CYBERSECURITY_IFT_MAX_PROMPT_CHARS = 3000
CYBERSECURITY_IFT_MAX_ANSWER_CHARS = 6000  # answers reach ~5.9k chars; keep them whole where possible

def _cybersecurity_ift_to_prepped(row):
    """Render one raw Trendyol row into the SFT schema ({prompt, answer})."""
    return {"prompt": (row["user"] or "").strip()[:CYBERSECURITY_IFT_MAX_PROMPT_CHARS],
            "answer": (row["assistant"] or "").strip()[:CYBERSECURITY_IFT_MAX_ANSWER_CHARS]}


SECEVAL_CHOICES = ["A", "B", "C", "D"]
SECEVAL_INSTRUCTION = ("The following is a multiple-choice question about cybersecurity. One or more options may "
                       "be correct. Respond with only the letters of all correct options.")

def _seceval_is_scorable(row):
    """Keep only rows with 4 options and a gold made up solely of A-D letters (single or multi answer)."""
    answer = (row.get("answer") or "").strip().upper()
    return (row.get("choices") and len(row["choices"]) == len(SECEVAL_CHOICES)
            and answer and all(c in SECEVAL_CHOICES for c in answer))

def _seceval_render_question(row):
    """The instruction + question + rendered options block (shared by the generative and SFT renders)."""
    return f"{SECEVAL_INSTRUCTION}\n\n{row['question'].strip()}\n\n" + "\n".join(row["choices"])

def _seceval_to_prepped(row):
    """Render one raw SecEval row into the SFT schema ({prompt, answer}); GenerativeCollate carries the gold
    letters through as the reference the completion is scored against."""
    return {"prompt": _seceval_render_question(row), "answer": (row["answer"] or "").strip().upper()}

def _seceval_letters(text):
    """The set of A-D option letters at the start of a completion, parsed consecutively ('AB' -> {'A','B'}, 'A B C D' -> {'A'}, 'B is correct' -> {'B'})."""
    letters = []
    for ch in text.strip().upper():
        if ch not in SECEVAL_CHOICES:
            break
        if ch not in letters:
            letters.append(ch)
    return set(letters)

def _seceval_f1(pred_text, gold_text):
    """Set F1 between the predicted and gold option letters.

    F1 rather than exact match so a partly-correct selection gets partial credit: precision punishes naming
    extra options, recall punishes missing gold ones. An unparseable completion scores 0 rather than raising."""
    pred, gold = _seceval_letters(pred_text), set(gold_text.strip().upper())
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    precision, recall = tp / len(pred), tp / len(gold)
    return 2 * precision * recall / (precision + recall)

class PreppedQACollate:
    """Collate for prepped scored-candidate QA datasets."""
    def __init__(self, tokenizer, dataset_tag, answer_prefix=""):
        self.tokenizer = tokenizer
        self.dataset_tag = dataset_tag
        # Text appended to the assistant turn so the scored next token is the answer letter (instruct models
        # otherwise start with "The"/"To", so A/B/C/D lose the first-token argmax). Part of `prompt` (scoring
        # context) and precedes the gold label in `text` (training sees the same format). The trailing quote
        # forces a no-space continuation, so the model's next token matches the bare-letter cand_ids that
        # task_eval scores (a trailing space would instead make " A" the natural token, breaking the match).
        self.answer_prefix = answer_prefix

    def __call__(self, rows):
        batch = {"dataset": self.dataset_tag, "text": [], "prompt": [], "cand_labels": [], "gold_index": []}
        for row in rows:
            labels = list(row["cand_labels"])
            gold_index = labels.index(row["answer"])
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True) + self.answer_prefix
            batch["text"].append(prompt + labels[gold_index])
            batch["prompt"].append(prompt)
            batch["cand_labels"].append(labels)
            batch["gold_index"].append(gold_index)
        return batch

def _chemdata_to_prepped(row):
    """Render one raw AI4Chem/ChemData700K row into the prepped SFT schema ({prompt, answer}). Each row is an Alpaca-style {instruction, input, output} record (history is empty here);"""
    parts = [p.strip() for p in (row.get("instruction", ""), row.get("input", "")) if p and p.strip()]
    return {"prompt": "\n\n".join(parts), "answer": row["output"]}


def _igbo_translation_to_prepped(row):
    """Render one raw Tommy0201/igbo_to_english_split row into the SFT schema ({prompt, answer}): the source
    sentence prompts the model and it is trained/scored over the translation. TRANSLATE_IGBO_TO_ENG picks
    which side is which. The same render feeds the BLEU benchmark, whose `answer` is the reference
    translation, so the flag flips the benchmark's direction in lockstep."""
    src, tgt = ("igbo", "english") if TRANSLATE_IGBO_TO_ENG else ("english", "igbo")
    return {"prompt": f"{IGBO_TRANSLATION_INSTRUCTION}\n\n{row[src].strip()}", "answer": row[tgt].strip()}

class GenerativeCollate:
    """Collate for benchmarks scored by free-form generation."""
    def __init__(self, tokenizer, dataset_tag):
        self.tokenizer = tokenizer
        self.dataset_tag = dataset_tag

    def __call__(self, rows):
        batch = {"dataset": self.dataset_tag, "prompt": [], "answer": []}
        for row in rows:
            batch["prompt"].append(self.tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True))
            batch["answer"].append(row["answer"])
        return batch

class SFTCollate:
    """Collate for free-form SFT datasets (AI4Chem/ChemData700K) prepped to {prompt, answer}. `dataset` is one of SFT_DATASET_TAGS."""
    def __init__(self, tokenizer, dataset_tag='sft'):
        self.tokenizer = tokenizer
        self.dataset_tag = dataset_tag

    def __call__(self, rows):
        batch = {"dataset": self.dataset_tag, "text": [], "prompt": [], "prompt_len": []}
        for row in rows:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            # prompt_len uses the same tokenizer settings (add_special_tokens=False) as the consumers
            # so it lines up with where the answer begins inside the tokenized `text`.
            prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            batch["text"].append(prompt + row["answer"])
            batch["prompt"].append(prompt)
            batch["prompt_len"].append(prompt_len)
        return batch


@torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
def task_eval(model, tokenizer, test_dl, config, max_length=2048, log_prefix="Performance", dataset_name=""):
    """Accuracy on a closed-answer QA dataset"""
    model.eval()
    if config.activate_cl_method:
        cl_accum_state = get_cl_accum_state(model)
        set_cl_accum_off(model)
    
    total_correct = 0
    total_examples = 0
    total_nll = 0.0

    pbar = tqdm(total=len(test_dl), desc="Evaluating Task", leave=False)
    for batch in test_dl:
        enc = tokenizer(batch["prompt"], return_tensors="pt", padding="longest", padding_side="left", truncation=True, max_length=max_length, add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        next_logits = logits[:, -1, :].float()
        log_probs = torch.log_softmax(next_logits, dim=-1)

        for row_i, (cand_labels, gold_index) in enumerate(zip(batch["cand_labels"], batch["gold_index"])):
            cand_ids = [tokenizer(lab, add_special_tokens=False)["input_ids"][0] for lab in cand_labels]  # first token id of each candidate; argmax over just those tokens
            pred = int(next_logits[row_i, cand_ids].argmax().item())
            total_correct += int(pred == gold_index)
            total_examples += 1
            total_nll += -log_probs[row_i, cand_ids[gold_index]].item()

        if local_rank == 0:
            pbar.update(1)

    total_correct = all_reduce(total_correct, normalize=False)
    total_examples = all_reduce(total_examples, normalize=False)
    total_nll = all_reduce(total_nll, normalize=False)

    acc = total_correct / max(1, total_examples)
    ppl = math.exp(total_nll / max(1, total_examples))  # one scored token per example, so total_examples is the token count

    if local_rank == 0:
        logger.info(f"\n[EVAL] {dataset_name} accuracy: {acc:.4f} ({int(total_correct)}/{int(total_examples)}), PPL: {ppl:.2f}")
        wandb_log({f"{log_prefix}_accuracy/{dataset_name}": acc, f"{log_prefix}_ppl/{dataset_name}": ppl})

    model.train()
    if config.activate_cl_method:
        set_cl_accum_state(model, cl_accum_state)
    torch.cuda.empty_cache()

    return acc, ppl


@torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
def generative_eval(model, tokenizer, test_dl, config, max_length=2048, max_new_tokens=512, log_prefix="Performance", dataset_name=""):
    """Score a benchmark whose answers must be generated rather than picked from candidates."""
    model.eval()
    if config.activate_cl_method:
        cl_accum_state = get_cl_accum_state(model)
        set_cl_accum_off(model)

    total_correct = 0
    total_examples = 0
    hypotheses, references = [], []  # chrF only: the decoded text, scored as a corpus after the loop
    samples = []  # (completion, gold) from the first batch, logged below so a format mismatch is visible

    pbar = tqdm(total=len(test_dl), desc="Evaluating Generation", leave=False)
    for batch in test_dl:
        enc = tokenizer(batch["prompt"], return_tensors="pt", padding="longest", padding_side="left", truncation=True, max_length=max_length, add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            generated = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False,  # greedy: the eval must be deterministic
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Left padding means every row's continuation starts at the same offset: the prompt width.
        completions = tokenizer.batch_decode(generated[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        if not samples:  # first batch only
            samples = list(zip(batch["prompt"], completions, batch["answer"]))[:GENERATIVE_EVAL_NUM_SAMPLES]

        if is_igbo_translation_benchmark(dataset_name):
            hypotheses.extend(c.strip() for c in completions)
            references.extend(batch["answer"])
        elif is_seceval(dataset_name):  # set F1 over the generated option letters (golds may name several)
            for completion, gold in zip(completions, batch["answer"]):
                total_correct += _seceval_f1(completion, gold)  # a float in [0,1]: the mean below is mean F1
                total_examples += 1
        else:
            pass

        if local_rank == 0:
            pbar.update(1)

    if local_rank == 0 and samples:  # rank 0 only: every rank has its own shard, one is enough to spot a format mismatch
        logger.info(f"\n[EVAL SAMPLES] {dataset_name} ({len(samples)} of the first batch)")
        for i, (prompt, completion, gold) in enumerate(samples):
            logger.info(f"  [{i}] prompt tail: ...{prompt[-200:]!r}")
            logger.info(f"  [{i}] generated:   {completion!r}")
            logger.info(f"  [{i}] gold:        {gold!r}")

    if is_igbo_translation_benchmark(dataset_name):
        # Gather the strings themselves (not a per-rank sum) so the corpus metric is scored over the full corpus exactly once.
        hypotheses = all_gather_list(hypotheses)
        references = all_gather_list(references)
        metric_name, metric = 'chrf', sacrebleu.corpus_chrf(hypotheses, [references])
        acc = metric.score / 100  # sacreBLEU reports 0-100; normalize to 0-1 to match the accuracy metrics
        if local_rank == 0:
            logger.info(f"\n[EVAL] {dataset_name} {metric_name}: {acc:.4f} ({len(hypotheses)} sentences)")
            wandb_log({f"{log_prefix}_{metric_name}/{dataset_name}": acc})
    else:
        total_correct = all_reduce(total_correct, normalize=False)
        total_examples = all_reduce(total_examples, normalize=False)
        acc = total_correct / max(1, total_examples)
        if local_rank == 0:
            if is_seceval(dataset_name):  # a mean over per-example F1 floats, not a hit count
                logger.info(f"\n[EVAL] {dataset_name} mean set F1: {acc:.4f} (over {int(total_examples)} questions)")
            else:
                logger.info(f"\n[EVAL] {dataset_name} generative accuracy: {acc:.4f} ({int(total_correct)}/{int(total_examples)})")
            wandb_log({f"{log_prefix}_accuracy/{dataset_name}": acc})

    model.train()
    if config.activate_cl_method:
        set_cl_accum_state(model, cl_accum_state)
    torch.cuda.empty_cache()

    return acc, float('nan')  # no ppl: there is no closed candidate set to score


def eval_dataset(model, tokenizer, dataloaders, config, max_length=2048, log_prefix="Performance", dataset_name=""):
    """Eval one dataset, fetching whatever loaders it needs from dataloaders ({name: (train_dl, test_dl, train_size)}). Returns (acc, ppl)."""
    # Closed-candidate benchmarks are scored by argmax over the candidates; generative ones must be decoded.
    def _is_generative(name):
        return is_igbo_translation_benchmark(name) or is_seceval(name)
    accuracy_eval = generative_eval if _is_generative(dataset_name) or _is_generative(DATASET_BENCHMARK.get(dataset_name)) else task_eval

    # Only generative_eval generates, so only it takes a generation budget (task_eval has no such argument).
    gen_kwargs = lambda name: {"max_new_tokens": DATASET_MAX_GEN_LENGTH[name]} if accuracy_eval is generative_eval and name in DATASET_MAX_GEN_LENGTH else {}

    test_dl = dataloaders[dataset_name][1]
    if is_sft(dataset_name):
        ppl = ppl_eval(model, tokenizer, test_dl, config, max_length=max_length, log_prefix=log_prefix, dataset_name=dataset_name)
        acc = float('nan')
        if dataset_name in DATASET_BENCHMARK:
            benchmark = DATASET_BENCHMARK[dataset_name]
            acc, _ = accuracy_eval(model, tokenizer, dataloaders[benchmark][1], config, max_length=max_length, log_prefix=log_prefix, dataset_name=benchmark, **gen_kwargs(benchmark))
        return acc, ppl
    return accuracy_eval(model, tokenizer, test_dl, config, max_length=max_length, log_prefix=log_prefix, dataset_name=dataset_name, **gen_kwargs(dataset_name))


"""Used in continual learning - fills in the "zero-shot" column from pre-computed scores instead of recomputing them."""
def reuse_initial_eval_scores(config, cross_tasks, log_prefix, initial_evaluation_scores):
    cur_lm_eval = initial_evaluation_scores["lm_eval"]
    cur_task_eval = initial_evaluation_scores["task_evals"][config.dataset]
    cur_eval_ppl = initial_evaluation_scores["ppls"][config.dataset]

    epoch_lm_evals = {k: [v] for k, v in cur_lm_eval.items()}  # {"task/metric": [zero_shot_score, score_epoch0, ...], ..., "total": [...]}
    epoch_ppls, epoch_task_evals = [cur_eval_ppl], [cur_task_eval]  # index 0 is the zero-shot eval
    cross_task_epoch_ppls = {name: [initial_evaluation_scores["ppls"][name]] for name in cross_tasks}  # {dataset_name: [zero_shot_ppl, ppl_epoch0, ...]}
    cross_task_epoch_evals = {name: [initial_evaluation_scores["task_evals"][name]] for name in cross_tasks}  # {dataset_name: [zero_shot_eval, eval_epoch0, ...]}
    # mean task accuracy over all datasets evaluated at this point (current + cross). index 0 is the zero-shot eval
    epoch_task_evals_total = [np.mean([cur_task_eval] + [v[-1] for v in cross_task_epoch_evals.values()]).tolist()]
    if local_rank == 0:
        wandb_log({f"{log_prefix}_accuracy/total": epoch_task_evals_total[-1]})

    return epoch_lm_evals, epoch_ppls, epoch_task_evals, cross_task_epoch_ppls, cross_task_epoch_evals, epoch_task_evals_total


class HFWithExternalCache(HFLM):
    @property
    def max_gen_toks(self):
        return 1024
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pretrained = self.pretrained.module if hasattr(self.pretrained, "module") else self.pretrained
        self.num_layers = pretrained.config.num_hidden_layers
        self.attention_type = pretrained.config._attn_implementation

    # used for generate_until / few generative tasks
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        from transformers.cache_utils import DynamicCache
        generation_kwargs["past_key_values"] = DynamicCache()  # standard cache for quadratic attention
        
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


def base_model_performance_cache_path(config):
    """Path for the cached base-model zero-shot lm-eval scores + captured generations, keyed by base
    model + task set (the two things those depend on). "" if caching is disabled."""
    if not config.base_model_performance_cache_dir:
        return ""
    model_short = config.model_name.split('/')[-1]
    tasks_key = "_".join(sorted(config.lm_eval_tasks))
    # The limit changes the scores, so it belongs in the key.
    limit = getattr(config, 'lm_eval_limit', None)
    limit_key = "" if limit is None else f"_limit_{limit}"
    return os.path.join(config.base_model_performance_cache_dir, f"{model_short}_{tasks_key}{limit_key}.json")


def lm_eval_harness(model, tokenizer, tasks, limit=None, is_sample=False, log_prefix="Evaluations", return_dummy=False, use_base_model_performance_cache=False):
    """
    Run evaluation on the specified tasks using the provided model and tokenizer.
    Will temporarily replace the model's attention layer with the mixer layer for evaluation.
    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use for encoding inputs.
        tasks: List of tasks to evaluate on.
        num_fewshot: Number of few-shot examples to use.
        batch_size: Batch size for evaluation.
        limit: Optional limit on the number of samples to evaluate.
        sample: allows sampling in generative tasks, currently set to False.
    Returns:
        results: Evaluation results for the specified tasks.
    """
    # task_to_num_fewshot = {"mmlu": 5, "hellaswag": 10, "arc_challenge": 25, "winogrande": 5, "gsm8k": 4} # qwen paper config
    
    # None => use the value baked into the task's YAML (humaneval/ifeval are 0-shot; mbpp is 3-shot with a custom first_n sampler that breaks if we override num_fewshot).
    task_to_num_fewshot = {"gsm8k": 0, "humaneval": None, "ifeval": None}
    task_to_batch_size = {"gsm8k": 256, "humaneval": 256, "ifeval": 541} # ifeval - full dataset
    # For each generative task, the single result metric to keep (the value lm-eval reports the score under, before the ",filter" suffix). gsm8k is handled separately (it has two extract filters).
    task_to_metric = {"humaneval": "pass@1", "ifeval": "prompt_level_strict_acc"}
    
    if tasks == []:
        return

    # Use cached zero-shot scores and generations (identical in all runs). Saved per base model x pt tasks combination
    base_model_performance_cache = base_model_performance_cache_path(config) if use_base_model_performance_cache else ""
    if base_model_performance_cache and os.path.exists(base_model_performance_cache):
        with open(base_model_performance_cache) as f:
            cached = json.load(f)
        results, full_text = cached["results"], cached["full_text"]
        if local_rank == 0:
            logger.info(f"Loaded cached zero-shot lm-eval scores ({sum(len(v) for v in full_text.values())} full-text samples over {len(full_text)} benchmarks) from {base_model_performance_cache}")
            wandb_log_data = {f"{log_prefix}/{k}": v for k, v in results.items()}
            logger.info(wandb_log_data)
            wandb_log(wandb_log_data)
        return results.get("total", 0), results, full_text

    # Return zero-valued results with the same shape as a real run (so epoch_lm_evals keys stay
    # consistent) without running any evaluation. Used to skip lm_eval on all but the last epoch.
    if return_dummy:
        results = {task: 0 for task in tasks} | {"total": 0}
        if local_rank == 0:
            why = "lm_eval_compute_last_epoch_only" if config.lm_eval_compute_last_epoch_only else "skip_lm_eval"
            logger.warning(f"[lm_eval] skipped ({why}=True) and no cached scores at "
                           f"{base_model_performance_cache or '<caching disabled>'}; reporting zeros for {tasks} "
                           f"-- these are placeholders, not measured scores")
        return 0, results, []

    captured_texts = {}  # {benchmark: [{"prompt", "resp", "filtered_resp"} per sample, capped per task]}

    model.eval()
    if config.activate_cl_method:
        cl_accum_state = get_cl_accum_state(model)
        set_cl_accum_off(model)
    
    def wrap_model(batch_size):
        if world_size == 1:
            wrapped_model = HFWithExternalCache(
                        pretrained=model.to(torch.bfloat16), tokenizer=tokenizer, max_length=8192, backend="causal", batch_size=batch_size, add_bos_token=False, dtype=torch.bfloat16)
        else:
            wrapped_model = HFWithExternalCache(
                        pretrained=model.module.to(torch.bfloat16), tokenizer=tokenizer, max_length=8192, backend="causal", batch_size=batch_size, add_bos_token=False, dtype=torch.bfloat16)
        return wrapped_model

    wandb_log_data = {}
    wrapped_model = None
    for i_task, task in enumerate(tasks):
        logger.info(f"\n[{task}] Starting evaluation\n")
        task_bs = task_to_batch_size[task]
        task_num_fewshot = task_to_num_fewshot[task]
        apply_chat_template = task not in ['mbpp', 'humaneval']
        if i_task % world_size == local_rank:
            with torch.no_grad():
                wrapped_model = wrap_model(batch_size=task_bs)
                lm_eval_output = simple_evaluate(
                    model=wrapped_model, limit=limit, tasks=[task],
                    num_fewshot=task_num_fewshot, device="cuda",
                    apply_chat_template=apply_chat_template, fewshot_as_multiturn=True,
                    log_samples=True, batch_size=task_bs, verbosity="ERROR", cache_requests=True, confirm_run_unsafe_code=True,
                )
                results = lm_eval_output['results'][task]
                captured_texts[task] = []
                
                _meta_keys = {'doc', 'doc_id', 'doc_hash', 'prompt_hash', 'target', 'target_hash',
                              'arguments', 'resps', 'filtered_resps', 'filter'}
                for s in lm_eval_output['samples'][task]:
                    if task == "gsm8k" and s.get('filter') == "strict-match": # gsm8k's reported score is flexible-extract
                        continue
                    captured_texts[task].append({
                        "doc_id": s.get('doc_id'),  # index into the benchmark, so samples can be joined across runs
                        "filter": s.get('filter'),  # which filter this row's filtered_resp/score came from
                        "prompt": s['arguments'][0][0] if s.get('arguments') else '',  # rendered prompt, chat template included
                        "resp": s['resps'][0][0] if s.get('resps') else '',
                        "filtered_resp": s['filtered_resps'][0] if s.get('filtered_resps') else '',
                        "target": s.get('target'),
                        "scores": {k: v for k, v in s.items() if k not in _meta_keys},  # per-sample pass/fail
                    })

            results = {k: v for k, v in results.items() if "std" not in k}
            results.pop('alias')
            if task == "gsm8k":
                # Keep only the relaxed (flexible-extract) filter; instruct models rarely emit the strict "#### N" format that strict-match requires.
                results = {k.split(",")[0]: v for k, v in results.items()
                           if k.endswith("flexible-extract")}
            elif task in task_to_metric:
                # Generative tasks report one or more metrics keyed as "<metric>,<filter>". Keep only the metric we care about.
                metric = task_to_metric[task]
                results = {k.split(",")[0]: v for k, v in results.items()
                           if k.split(",")[0] == metric}
            else:
                results = {k.split(",")[0]: v for k, v in results.items()}
                # For tasks reporting both acc and acc_norm, prefer the length-normalized accuracy and drop the unnormalized one.
                if "acc_norm" in results:
                    results.pop("acc", None)
            
            wandb_log_data = wandb_log_data | {f"{log_prefix}/{task}": v for k,v in results.items()}
            logger.info(f"\n[{task}] Evaluation results: {[f'{k}={v:.2f}' for k,v in results.items()]}\n")
            print(f"Rank {local_rank}: results after update {wandb_log_data}")

    if wrapped_model is None:  # this avoids a deadlock in the multi-gpu case
        wrap_model(batch_size=1)

    barrier()
    if world_size > 1:
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, wandb_log_data)
        gathered_texts = [None] * world_size  # tasks are sharded across ranks, so each rank captured its own (disjoint) benchmarks' texts
        torch.distributed.all_gather_object(gathered_texts, captured_texts)
        captured_texts = {task: texts for shard in gathered_texts for task, texts in shard.items()}
    else:
        gathered = [wandb_log_data]

    wandb_log_data = {k: v for d in gathered for k, v in d.items()}
    total_score = np.mean(list(wandb_log_data.values())).tolist()
    wandb_log_data = wandb_log_data | {f"{log_prefix}/total": total_score}
    if local_rank == 0:
        logger.info(wandb_log_data)
        wandb_log(wandb_log_data)

    model.train()
    if config.activate_cl_method:
        set_cl_accum_state(model, cl_accum_state)
    torch.cuda.empty_cache()

    if world_size == 1:
        model = model.to(torch.float32)
    else:
        model.module = model.module.to(torch.float32)

    barrier()

    # prefix-free results for collection into summaries: {"{task}": v, ..., "total": total_score}
    results = {k[len(log_prefix) + 1:]: v for k, v in wandb_log_data.items()}

    if not return_dummy and local_rank == 0:
        samples_dir = f'./output/{config.start_time}_{config.wandb_run_name}/lm_eval_samples'
        os.makedirs(samples_dir, exist_ok=True)
        # log_prefix alone repeats every epoch, so number the files in call order rather than overwriting.
        lm_eval_harness._save_idx = getattr(lm_eval_harness, "_save_idx", 0) + 1
        samples_path = os.path.join(samples_dir, f"{log_prefix}_{lm_eval_harness._save_idx:03d}_samples.json")
        with open(samples_path, 'w') as f:
            json.dump({"results": results, "samples": captured_texts}, f, indent=1, default=str)
        logger.info(f"Saved per-sample lm-eval record ({sum(len(v) for v in captured_texts.values())} samples over {len(captured_texts)} benchmarks) to {samples_path}")

    if base_model_performance_cache and not return_dummy and local_rank == 0:  # cache for reuse; never cache the dummy (all-zero) results
        os.makedirs(os.path.dirname(base_model_performance_cache) or ".", exist_ok=True)
        with open(base_model_performance_cache, 'w') as f:
            json.dump({"results": results, "full_text": captured_texts}, f)
        logger.info(f"Cached zero-shot lm-eval scores ({sum(len(v) for v in captured_texts.values())} full-text samples over {len(captured_texts)} benchmarks) to {base_model_performance_cache}")

    return total_score, results, captured_texts

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    tokenizer.truncation_side = "left"  # keep the answer at the end of the sequence (right-truncation would drop the label)
    return tokenizer

def load_model(config):
    model = Qwen2ForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map=local_rank,
        attn_implementation="flash_attention_2",
    )

    model = model.to(torch.float32)
    if config.use_grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if config.use_conditional_lora:
        jl_k = config.jl_k if config.activate_jl else 0  # 0 disables JL
        apply_conditional_lora_wrapper(model, is_cl_lora=config.activate_cl_method, jl_k=jl_k, jl_seed=config.seed, cl_memory_type=config.cl_memory_type, smoothing_alpha=config.smoothing_alpha)
        if local_rank == 0:
            logger.info(f"Conditional LoRA wrapper applied (r={config.lora_r}, alpha={config.lora_alpha}); base frozen, K=0.")
        
        if config.load_cl_state_path:  # reload a saved cl state (adapters + cl_memory) into the fresh K=0 wrapper
            load_cl_state(model, config.load_cl_state_path)
            if local_rank == 0:
                logger.info(f"Loaded cl state from {config.load_cl_state_path}")

    if local_rank == 0:
        total_layers_params, total_other_params = get_param_count(model)
        logger.info(f"Total param count: {(total_layers_params+total_other_params) / 1e6:.2f}M \nLayers: {total_layers_params / 1e6:.2f}M, Other: {total_other_params / 1e6:.2f}M")

    return model


def run_continual_learning_evals(model, tokenizer, pretrain_config, dataloaders, pretrain_ppl=99999):
    """
    Continual learning evaluation: finetune on tasks sequentially (without resetting weights between tasks).
    After every epoch of each task, evaluate on ALL tasks.
    dataloaders: {dataset: (multi_epoch_train_dl, test_dl, train_size)}.
    """
    # Under LoRA the model isn't DDP-wrapped yet (main() wraps it per task), so unwrap defensively.
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    # With conditional LoRA the base is frozen and each task's adapter persists, so we
    # don't snapshot/restore pretrain weights between tasks.
    # pretrain_state = copy.deepcopy(raw_model.state_dict())

    continual_learning_datasets = pretrain_config.continual_learning_datasets

    ppl_results = {}  # key: (task_idx, dataset_name) -> [ppl_epoch0, ppl_epoch1, ...]
    task_eval_results = {}  # key: (task_idx, dataset_name) -> [acc_epoch0, acc_epoch1, ...]
    lm_eval_results = {}  # key: task_idx -> {"task": [zero_shot, epoch0, ...], ..., "total": [...]}

    if local_rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"[CONTINUAL LEARNING EVAL] Starting (pretrain PPL={pretrain_ppl:.2f})")
        logger.info(f"  Task order: {' -> '.join(continual_learning_datasets)}")
        logger.info(f"{'='*60}")

    training_phases = continual_learning_datasets

    wandb_step = 0
    initial_evaluation_scores = None  # task 0 computes its own zero-shot eval; tasks 1+ reuse the previous task's final eval
    for task_idx, cl_dataset in enumerate(training_phases):
        cl_config = copy.deepcopy(pretrain_config)
        cl_config.dataset = cl_dataset
        cl_config.task_idx = task_idx  # which CL phase this is; main() fits the negative GMMs on phase 0 only
        cl_config.ft_max_num_epochs = getattr(pretrain_config, 'ft_epochs_per_dataset', {}).get(cl_dataset, pretrain_config.ft_max_num_epochs) # Per-dataset epoch count; datasets absent from the dict keep the global ft_max_num_epochs.
        cl_config.wandb_logger = pretrain_config.wandb_logger
        cl_config.save_model_during_training = False
        cl_config.eval_only = False
        cl_config.num_eval_steps = torch.inf # eval only when epoch ends
        cl_config.logger_step_size = 1 #20
        cl_config.eval_ppl_checkpoints = ()
        # Evaluate on all other tasks (the current training task is evaluated by main). For IID there is no single current task, so every CL dataset is a cross-eval.
        cl_config.eval_datasets = [ds for ds in continual_learning_datasets if ds != cl_dataset]

        if local_rank == 0:
            logger.info(f"\n[CONTINUAL LEARNING EVAL] Task {task_idx+1}/{len(training_phases)}: {cl_dataset}")
            task_order_str = ' -> '.join(f"***{ds}***" if i == task_idx else ds for i, ds in enumerate(training_phases))
            logger.info(f"  Task order: {task_order_str}")

        # A loaded cl_state already re-added the LoRAs
        needs_lora = not cl_config.load_cl_state_path and (task_idx == 0 or (task_idx > 0 and cl_config.allocate_lora_per_phase))
        if cl_config.use_conditional_lora and needs_lora:
            add_lora(raw_model, r=cl_config.lora_r, alpha=cl_config.lora_alpha, gmm_kwargs=build_gmm_kwargs(cl_config))
            if local_rank == 0:
                num_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                logger.info(f"Added LoRA for task {task_idx+1}; trainable params: {num_trainable / 1e6:.2f}M")

        # Note: we do NOT reset model weights between tasks
        tasks = {"current": cl_dataset, "cross": cl_config.eval_datasets}
        epoch_ppls, epoch_task_evals, epoch_task_evals_total, cross_task_epoch_ppls, cross_task_epoch_evals, epoch_lm_evals, wandb_step = main(cl_config, model, tokenizer, dataloaders, tasks, log_prefix="CL", wandb_step=wandb_step, initial_evaluation_scores=initial_evaluation_scores)
        ppl_results[(task_idx, cl_dataset)] = epoch_ppls
        task_eval_results[(task_idx, cl_dataset)] = epoch_task_evals
        task_eval_results[(task_idx, 'total')] = epoch_task_evals_total
        for cross_ds, cross_ppls in cross_task_epoch_ppls.items():
            ppl_results[(task_idx, cross_ds)] = cross_ppls
        for cross_ds, cross_task_evals in cross_task_epoch_evals.items():
            task_eval_results[(task_idx, cross_ds)] = cross_task_evals
        lm_eval_results[task_idx] = epoch_lm_evals

        # Collect this phase's final-epoch scores so the next phase can reuse them as its zero-shot eval.
        initial_evaluation_scores = {
            "task_evals": {cl_dataset: epoch_task_evals[-1], **{ds: vals[-1] for ds, vals in cross_task_epoch_evals.items()}},
            "ppls": {cl_dataset: epoch_ppls[-1], **{ds: vals[-1] for ds, vals in cross_task_epoch_ppls.items()}},
            "lm_eval": {k: vals[-1] for k, vals in epoch_lm_evals.items()},
        }

    if local_rank == 0:
        save_dir = f'./results/{pretrain_config.start_time}_{pretrain_config.wandb_run_name}_best'
        log_continual_learning_eval_summary(continual_learning_datasets, ppl_results, task_eval_results, lm_eval_results, pretrain_ppl, save_dir)

'''
Training loop code. Can run in two modes:
 - Main loop (*** currently not used ***): Regular training (pretrainig \ finetuning)
 - Fintune ppl checkpoint (evaluations that require finetuning):
   Implemented by a recursive call from the main loop (run_continual_learning_evals).
   Finetune the current pre-trained model for evaluation and then resumes pretraining from the same state.
'''
def main(config, model, tokenizer, dataloaders, tasks, log_prefix="Performance", wandb_step=0, initial_evaluation_scores=None):

    current_task            = tasks["current"]
    cross_tasks             = tasks["cross"]
    train_dl, _, train_size = dataloaders[current_task]  # eval loaders are fetched by name in eval_dataset
    output_dir              = f'./output/{config.start_time}_{config.wandb_run_name}'

    # Set up hyperparameters
    global_batch_size           = config.ft_global_batch_size # batch size across all GPUs and gradient accumulation steps
    local_batch_size            = config.local_batch_size  # batch size per GPU per gradient accumulation step
    gradient_accumulation_steps = global_batch_size // (local_batch_size * world_size)
    learning_rate               = config.ft_learning_rate
    min_learning_rate           = config.ft_min_learning_rate
    max_num_epochs              = config.ft_max_num_epochs

    assert global_batch_size % (local_batch_size * world_size) == 0, f"global_batch_size {global_batch_size} must be divisible by local_batch_size {local_batch_size} * world_size {world_size}={local_batch_size * world_size}"

    if local_rank == 0:
        logger.info(f'Dataset: {config.dataset}')

    if use_ddp:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = wrap_model_ddp(model)
    else:
        if not next(model.parameters()).is_cuda:
            model = model.to('cuda')

    steps_per_epoch_per_rank    = math.ceil(train_size / (local_batch_size * gradient_accumulation_steps))  # train_size is per-rank (already sharded)
    num_update_steps            = max_num_epochs * steps_per_epoch_per_rank
    config.warmup_ratio         = config.ft_warmup_ratio
    config.stable_ratio         = config.ft_stable_ratio
    config.decay_ratio          = config.ft_decay_ratio
    warmup_steps                = int(num_update_steps * config.warmup_ratio)
    stable_steps                = int(num_update_steps * config.stable_ratio)  # 0.48
    decay_steps                 = int(num_update_steps * config.decay_ratio)  # 0.48
    assert config.warmup_ratio + config.stable_ratio + config.decay_ratio == 1.0, f"wsd ratios do not sum to 1 (sum = {config.warmup_ratio + config.stable_ratio + config.decay_ratio})"

    if config.eval_only:
        if not config.skip_lm_eval:
            lm_eval_harness(model, tokenizer, config.lm_eval_tasks, limit=config.lm_eval_limit)
        eval_dataset(model, tokenizer, dataloaders, config, max_length=config.seq_len_eval, log_prefix=log_prefix, dataset_name=current_task)
        return wandb_step

    model.train()
    if config.activate_cl_method and config.cl_memory_type == 'uob':
        set_cl_accum_on(model)
    
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(nd in n.lower() for nd in ["bias", "norm"]):
                no_decay.append(p)
            else:
                decay.append(p)

    if use_ddp:
        optimizer = ZeroRedundancyOptimizer(
            [{"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}], optimizer_class=torch.optim.AdamW,
            lr=learning_rate, betas=(config.beta1, config.beta2),
            )
    else:
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate, betas=(config.beta1, config.beta2)
        )

    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        num_training_steps=num_update_steps,
        num_stable_steps=stable_steps,
        warmup_type="linear",
        decay_type="cosine",
        min_lr_ratio = min_learning_rate / learning_rate,
        num_cycles=0.5,
    )

    # eval initial model
    if initial_evaluation_scores is not None:  # CL task N>0: reuse the previous task's final eval (identical model state) instead of recomputing
        epoch_lm_evals, epoch_ppls, epoch_task_evals, cross_task_epoch_ppls, cross_task_epoch_evals, epoch_task_evals_total = reuse_initial_eval_scores(config, cross_tasks, log_prefix, initial_evaluation_scores)
    else:
        cross_task_epoch_ppls, cross_task_epoch_evals = {name: [] for name in cross_tasks}, {name: [] for name in cross_tasks} # {dataset_name: [zero_shot, epoch0, ...]}
        return_dummy = config.lm_eval_compute_last_epoch_only or config.skip_lm_eval # skip_lm_eval still consults the base-model cache (free); on a miss it falls back to dummy results rather than running the evals.
        _, cur_lm_eval, _ = lm_eval_harness(model, tokenizer, config.lm_eval_tasks, limit=config.lm_eval_limit, return_dummy=return_dummy, use_base_model_performance_cache=True)
        epoch_lm_evals = {k: [v] for k, v in cur_lm_eval.items()}  # {"task/metric": [zero_shot_score, score_epoch0, ...], ..., "total": [...]}
        for eval_ds_name in cross_tasks:
            cross_eval, cross_ppl = eval_dataset(model, tokenizer, dataloaders, config, max_length=config.seq_len_eval, log_prefix=log_prefix, dataset_name=eval_ds_name)
            cross_task_epoch_ppls[eval_ds_name].append(cross_ppl)
            cross_task_epoch_evals[eval_ds_name].append(cross_eval)

        cur_task_eval, cur_eval_ppl = eval_dataset(model, tokenizer, dataloaders, config, max_length=config.seq_len_eval, log_prefix=log_prefix, dataset_name=current_task)
        # mean task accuracy over all datasets evaluated at this point (current + cross). index 0 is the zero-shot eval
        epoch_task_evals_total = [np.mean([cur_task_eval] + [v[-1] for v in cross_task_epoch_evals.values()]).tolist()]
        epoch_ppls, epoch_task_evals = [cur_eval_ppl], [cur_task_eval]  # index 0 is the zero-shot eval
        if local_rank == 0:
            wandb_log({f"{log_prefix}_accuracy/total": epoch_task_evals_total[-1]})

    # Negative GMMs: fit (or load) after the zero-shot eval. The zero-shot eval should run before the GMMs start gating (mirrors uob, whose memory is likewise empty at zero-shot). 
    # The positive GMMs are refitted after each epoch below.
    # The negative GMM describes a fixed generic corpus, so it is phase-independent: fit it once on phase 0
    # and let later phases reuse it (add_lora shares the existing neg_gmm into each new phase's classifier).
    task_idx = getattr(config, 'task_idx', 0)  # absent for the finetune (TL) path, which is single-phase
    if config.activate_cl_method and config.cl_memory_type == 'gmm':
        if config.train_neg_gmm and task_idx == 0:
            if local_rank == 0:
                logger.info("Fitting negative cl_memory GMMs ...")
            active_ratio = fit_cl_memory(model, dataloaders['gmm_neg'][0], tokenizer, config, which='neg')
            if local_rank == 0:
                wandb_log({"gmm_eval/active_owners_ratio": active_ratio})
                save_path = f'{output_dir}/neg_gmms.pt'
                save_gmms(model, save_path, which='neg')
                logger.info(f"Saved negative GMMs to {save_path}")
        elif config.train_neg_gmm:
            if local_rank == 0:
                logger.info(f"Reusing the negative cl_memory GMMs fitted on phase 0 (phase {task_idx})")
        else:
            if local_rank == 0:
                logger.info(f"Loading negative cl_memory GMMs from {config.gmm_neg_path} ...")
            load_gmms(model, config.gmm_neg_path, which='neg')

    _seen_tokens = 0.0
    best_ppl = torch.inf
    ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    current_epoch = 0
    for global_step in range(num_update_steps):
        optimizer.zero_grad(set_to_none=True)
        train_loss_global_step = 0.0
        aux_loss_global_step = 0.0
        total_loss_global_step = 0.0
        _seen_tokens_global_step = 0

        mini_batches = []
        for _ in range(gradient_accumulation_steps):
            cur_mb, epoch_ended = next(train_dl)
            cur_mb_input = tokenizer(cur_mb["text"], return_tensors="pt", max_length=config.seq_len_train,padding="longest",padding_side="left",truncation=True, add_special_tokens=False).to("cuda")
            cur_mb_input_ids = cur_mb_input.input_ids
            cur_mb_attention_mask = cur_mb_input.attention_mask
            cur_mb_labels = cur_mb_input_ids.clone()
            cur_mb_labels[cur_mb_labels == tokenizer.pad_token_id] = -100  # ignore pad tokens by mask
            if cur_mb.get("dataset") in QA_DATASET_TAGS:                    # ignore prompt by mask for QA finetuning
                cur_mb_labels[:, :-1] = -100
            elif cur_mb.get("dataset") in SFT_DATASET_TAGS:                 # SFT: mask each row's prompt span, train on the full answer
                seq_len = cur_mb_labels.shape[1]
                for row_i, plen in enumerate(cur_mb["prompt_len"]):
                    pad_offset = int((cur_mb_attention_mask[row_i] == 0).sum().item())  # left pads precede the (truncated) text
                    cur_mb_labels[row_i, :min(pad_offset + plen, seq_len)] = -100
            cur_mb_shift_labels = torch.hstack([cur_mb_labels[:,1:],torch.full([cur_mb_labels.shape[0],1], -100, device=cur_mb_labels.device)])
            num_tokens_in_mb = (cur_mb_shift_labels != -100).sum().item()
            _seen_tokens_global_step += num_tokens_in_mb

            mini_batches.append((cur_mb, cur_mb_input_ids, cur_mb_shift_labels, cur_mb_attention_mask))
            if epoch_ended:
                break

        for cur_mb, cur_mb_input_ids, cur_mb_shift_labels, cur_mb_attention_mask in mini_batches:
            mb_train_loss = 0.0
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(cur_mb_input_ids.contiguous(), attention_mask=cur_mb_attention_mask.contiguous())
                flat_logits = output.logits.view([-1, output.logits.shape[-1]])
                flat_labels = cur_mb_shift_labels.contiguous().view(-1)
                scored = flat_labels != -100
                cur_loss = ce_loss(flat_logits[scored].float(), flat_labels[scored])  # sum of nlls per chunk   # convert logits to float for safe computation 
                mb_train_loss += cur_loss

            mb_train_loss = mb_train_loss / _seen_tokens_global_step
            aux_loss     = torch.tensor(0, device=mb_train_loss.device)
            loss         = mb_train_loss + aux_loss

            # metrics
            train_loss_global_step += mb_train_loss.detach().float().item()
            aux_loss_global_step += aux_loss.detach().float().item()
            total_loss_global_step += loss.detach().float().item()

            loss.backward()
        
        if config.activate_cl_method and config.cl_memory_type == 'uob' and local_rank == 0:
            report_cl_memory(model)

        # optimizer & scheduler step
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        total_grad_norm = min(total_grad_norm, config.grad_clip_norm)
        total_reg_grad_norms_sqrd, total_other_grad_norms_sqrd, reg_grad_norms_per_layer = get_grad_norms(model)
        
        # sync among ranks.
        # we normalize here because we intentionally compute the loss per gpu (this way when .backward() does all_reduce, it has the correct grad (over the whole batch), and here we get the correct loss for the metric)
        train_loss_global_step                            = all_reduce(train_loss_global_step, normalize=True)
        aux_loss_global_step                              = all_reduce(aux_loss_global_step, normalize=True)
        total_loss_global_step                            = all_reduce(total_loss_global_step, normalize=True)
        _seen_tokens_global_step                          = all_reduce(_seen_tokens_global_step, normalize=False)
        total_grad_norm                                   = all_reduce(total_grad_norm, normalize=True)
        reg_grad_norms_per_layer                          = all_reduce(reg_grad_norms_per_layer, normalize=True)
        total_reg_grad_norms_sqrd                         = all_reduce(total_reg_grad_norms_sqrd, normalize=True)
        total_other_grad_norms_sqrd                       = all_reduce(total_other_grad_norms_sqrd, normalize=True)
        epoch_ended                                       = all_reduce(int(epoch_ended)) > 0  # Max via sum. All ranks should finish the epoch when one is finished 

        _seen_tokens += _seen_tokens_global_step
        if local_rank == 0:
            wandb_log({"Train/train_loss": train_loss_global_step,
                        "Train/perplexity": np.exp(train_loss_global_step).tolist(),
                        "Train/aux_loss": aux_loss_global_step,
                        "Train/loss": total_loss_global_step,
                        "Train/lr": optimizer.param_groups[0]['lr'],
                        "Train/grad_norm": total_grad_norm,
                        "Train/grad_norm_sqrd": total_grad_norm**2,
                        "Train/reg_layers_grad_norm_sqrd": total_reg_grad_norms_sqrd,
                        "Train/other_layers_grad_norm_sqrd": total_other_grad_norms_sqrd,
                        "Train/seen_tokens": _seen_tokens,
                        "Train/epoch": current_epoch,
                        "Train/new_phase": 1 if global_step == 0 else 0,} | \
                        {f'Train/reg_grad_norm_layer_{i}': reg_grad_norms_per_layer[i].tolist() for i in range(reg_grad_norms_per_layer.shape[0])}
            )

        optimizer.step()
        scheduler.step()
        if config.activate_cl_method and config.cl_memory_type == 'uob':
            update_cl_memory(model)
            if local_rank == 0:
                report_compaction(model)
                bd = cl_memory_breakdown(model)
                wandb_log({"cl_memory/total_mb": bd["total_mb"]}
                          | {f"cl_memory/{t}_mb": mb for t, mb in bd["per_type"].items()}
                          | {f"cl_memory/L{l}_mb": mb for l, mb in bd["per_layer"].items()})

        if global_step % config.logger_step_size == 0 and local_rank == 0:
            avg_time = (time.time() - step_start_time)/config.logger_step_size if "step_start_time" in locals() else -1
            step_start_time = time.time()
            logger.info(f"\n[PROGRESS] Epoch {current_epoch}, Global Step {global_step}, PPL: {np.exp(train_loss_global_step):.4f}, Avg time: {avg_time:.2f} seconds per step")
        
        # Refit the positive GMMs on the epoch's updated representation, before the evals below use them.
        if epoch_ended and config.activate_cl_method and config.cl_memory_type == 'gmm':
            if local_rank == 0:
                logger.info(f"\n[PROGRESS] Fitting cl_memory GMMs at end of epoch {current_epoch}")
            active_ratio = fit_cl_memory(model, dataloaders[f'gmm_pos_{current_task}'][0], tokenizer, config, which='pos')
            if local_rank == 0:
                wandb_log({"gmm_eval/active_owners_ratio": active_ratio})
                save_path = f'{output_dir}/pos_gmms_{current_task}_phase{task_idx}_epoch_{current_epoch}.pt'
                save_gmms(model, save_path, which='pos')
                logger.info(f"Saved positive GMMs to {save_path}")


        # periodic evaluation
        if (global_step % config.num_eval_steps == 0 and global_step) or (global_step == num_update_steps-1) or epoch_ended:
            is_last_epoch = current_epoch == max_num_epochs-1 # assumes cl (config.num_eval_steps=inf)
            return_dummy = config.lm_eval_compute_last_epoch_only and not is_last_epoch
            pt_tasks_texts_cur = None
            if config.skip_lm_eval:
                cur_lm_eval = {}
            else:
                _, cur_lm_eval, pt_tasks_texts_cur = lm_eval_harness(model, tokenizer, config.lm_eval_tasks, limit=config.lm_eval_limit, return_dummy=return_dummy)

            if epoch_ended and config.activate_cl_method and config.cl_memory_type == 'gmm':
                pt_tasks_dls = {bench: make_text_dl(list(dict.fromkeys(s["prompt"] + s["resp"] for s in samples)), config.eval_batch_size)
                                for bench, samples in pt_tasks_texts_cur.items()} if pt_tasks_texts_cur else None
                gmm_scores = gmm_eval(model, dataloaders[current_task][1], dataloaders['gmm_neg'][1], tokenizer, config, pt_tasks_dls=pt_tasks_dls)
                if local_rank == 0:
                    report_gmm_eval(gmm_scores, logger=logger, wandb_log=wandb_log)

            cur_task_eval, cur_eval_ppl = eval_dataset(model, tokenizer, dataloaders, config, max_length=config.seq_len_eval, log_prefix=log_prefix, dataset_name=current_task)
            if epoch_ended:
                for k, v in cur_lm_eval.items():
                    epoch_lm_evals[k].append(v)

                # Cross-task evaluation for continual learning
                if cross_tasks:
                    for eval_ds_name in cross_tasks:
                        cross_eval, cross_ppl = eval_dataset(model, tokenizer, dataloaders, config, max_length=config.seq_len_eval, log_prefix=log_prefix, dataset_name=eval_ds_name)
                        cross_task_epoch_ppls[eval_ds_name].append(cross_ppl)
                        cross_task_epoch_evals[eval_ds_name].append(cross_eval)

                # mean task accuracy over all datasets evaluated at this point (current + cross)
                epoch_task_evals_total.append(np.mean([cur_task_eval] + [v[-1] for v in cross_task_epoch_evals.values()]).tolist())
                epoch_ppls.append(cur_eval_ppl)
                epoch_task_evals.append(cur_task_eval)
                if local_rank == 0:
                    wandb_log({f"{log_prefix}_accuracy/total": epoch_task_evals_total[-1]})

                if config.save_cl_state and config.activate_cl_method and local_rank == 0:
                    save_path = f'{output_dir}/cl_state_{current_task}_epoch_{current_epoch}.pt'
                    logger.info(f"\n[PROGRESS] Saving cl state to {save_path}")
                    save_cl_state(model, save_path)

            if cur_eval_ppl < best_ppl:
                if local_rank == 0:
                    logger.info(f"\n[PROGRESS] Found best ppl {cur_eval_ppl:.2f} (prev best = {best_ppl:.2f})")

                    if config.save_model_during_training:
                        logger.info(f"\n[PROGRESS] Saving best model")
                        save_dir = f'./output/{config.start_time}_{config.wandb_run_name}_best'
                        os.makedirs(save_dir, exist_ok=True)
                        filename = os.path.join(save_dir, f"model_weights.safetensors")
                        if use_ddp:
                            save_file(model.module.state_dict(), filename)
                        else:
                            save_file(model.state_dict(), filename)

                best_ppl = cur_eval_ppl

        if config.wandb_logger and local_rank == 0:
            assert wandb_step == wandb.run.step
            wandb_commit()
            wandb_step += 1

        if epoch_ended:
            current_epoch = current_epoch + 1
            if config.activate_cl_method and config.flush_cl_memory_after_each_epoch and current_epoch < max_num_epochs:
                if local_rank == 0:
                    logger.info(f"\n[PROGRESS] Flushing cl_memory at end of epoch {current_epoch - 1}")
                flush_cl_memory(model)

        if current_epoch == max_num_epochs:
            break

    return epoch_ppls, epoch_task_evals, epoch_task_evals_total, cross_task_epoch_ppls, cross_task_epoch_evals, epoch_lm_evals, wandb_step


if __name__ == "__main__":

    print(f"Rank {os.environ.get('LOCAL_RANK')} starting", flush=True)
    os.environ["WANDB_API_KEY"] = "" # YOUR-API-KEY

    
    mp.set_start_method('fork', force=True)

    
    from configs.config_continual import Configuration
    config = Configuration()
    logger.info('\n'.join(f"{k} = {v}" for k, v in config.__dict__.items()))
    start_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    config.start_time = start_time

    # We need to build the dataloaders before CUDA init so the fork-based workers spawn from a CUDA-clean process (multiprocessing_context='fork' deadlocks if forked after CUDA is initialized).
    tokenizer = load_tokenizer(config)
    dataloaders = build_dataloaders(config, tokenizer)

    # Initialize WandB
    with dist_group():
        with wandb_init(config, local_rank) as run:
            model = load_model(config)
            set_shared_prefix_ids(model, tokenizer) # Cache the chat template's shared prefix on the model (see set_shared_prefix_mask).
            # With conditional LoRA the base is frozen and no param requires grad until the
            # first add_lora (in the CL loop), so DDP can't wrap an empty module here. main()
            # re-wraps DDP per task after that task's add_lora, so the new LoRA params land in
            # DDP's reducer and sync across ranks.
            if use_ddp and not config.use_conditional_lora:
                if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model = wrap_model_ddp(model)
            else:
                if not next(model.parameters()).is_cuda:
                    model = model.to('cuda')

            barrier()
            if local_rank == 0:
                logger.info(f"All ranks were initialized.")

            torch.manual_seed(config.seed); random.seed(config.seed); np.random.seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            run_continual_learning_evals(model, tokenizer, config, dataloaders)