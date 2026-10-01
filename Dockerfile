# Cluster-recommended B200 base: PyTorch 2.9 + CUDA 13.0, NCCL/cuDNN pre-tuned.
FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# transformers pinned to the v4.56.2 tag (upstream). The local submodule has 3
# extra commits, but those are for a different model path and aren't needed here.
RUN git clone --depth 1 --branch v4.56.2 \
        https://github.com/huggingface/transformers.git /opt/transformers && \
    pip install --no-cache-dir /opt/transformers && \
    rm -rf /opt/transformers/.git

# Python deps (lm_eval is imported by train_continual.py)
RUN pip install --no-cache-dir \
    datasets \
    wandb \
    loguru \
    safetensors \
    tqdm \
    numpy \
    accelerate \
    tokenizers \
    sentencepiece \
    protobuf \
    langdetect \
    immutabledict \
    nltk \
    pomegranate \
    sacrebleu \
    tabulate \
    "lm_eval==0.4.10"

# ifeval scores responses with nltk.word_tokenize, which needs the punkt_tab
# tokenizer data at runtime (not pulled in by pip). Download into a path on
# nltk's default search list so every rank finds it regardless of HOME.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab punkt

# flash-attn must build after torch is present; --no-build-isolation reuses
# the base image's torch + CUDA toolchain.
RUN pip install --no-cache-dir flash-attn --no-build-isolation
