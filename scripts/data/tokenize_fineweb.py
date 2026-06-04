# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenize a FineWeb (or other text) parquet dataset into ``.bin`` / ``.idx`` shards.

This streams each parquet file directly (avoiding the HuggingFace datasets cache,
which keeps disk/inode usage low) and writes, for every input parquet file, one
pair of output files:

  * ``<prefix>_shard_<id>.bin`` -- all document token ids concatenated,
    stored as little-endian ``uint32`` (4 bytes per token).
  * ``<prefix>_shard_<id>.idx`` -- ``int64`` document-start offsets into the
    ``.bin`` file, plus a final trailing offset equal to the total token count.
    A document ``i`` therefore occupies tokens ``[idx[i], idx[i+1])``.

This is exactly the layout consumed by ``BinIdxDataset`` in
``torchtitan/datasets/hf_datasets.py``. Tokenization uses the Llama-2
SentencePiece tokenizer with BOS/EOS added per document.

Example:
    python scripts/data/tokenize_fineweb.py \\
        --input-glob "/path/to/fineweb/data/CC-MAIN-2023-50/*.parquet" \\
        --output-dir /path/to/fineweb_tokenized \\
        --tokenizer-path ./assets/tokenizer.model \\
        --shard-prefix fineweb_tokens \\
        --num-workers 32
"""

import argparse
import gc
import glob
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import datasets
import numpy as np

# This script lives in scripts/data/; add the repo root (two levels up) to
# sys.path so that `torchtitan` can be imported.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torchtitan.datasets import create_tokenizer

# Llama-2 uses the SentencePiece tokenizer (see model_name_to_tokenizer in
# torchtitan/models/__init__.py). Hardcoded here so this CPU-only preprocessing
# script does not import the model package (which pulls in Triton/CUDA).
TOKENIZER_TYPE = "sentencepiece"


def process_single_file(file_path, shard_id, output_dir, tokenizer_path, shard_prefix, log_file_path):
    """Tokenize one parquet file into a single .bin/.idx shard pair.

    Run inside a worker process, so the tokenizer is created locally here.
    """
    try:
        print(f"[pid {os.getpid()}] start {file_path}", flush=True)
        tokenizer = create_tokenizer(TOKENIZER_TYPE, tokenizer_path)
        ds = datasets.load_dataset("parquet", data_files=file_path)

        bin_path = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_id:05d}.bin")
        idx_path = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_id:05d}.idx")
        num_rows = len(ds["train"])
        print(f"[pid {os.getpid()}] writing {num_rows} rows to {bin_path}", flush=True)

        with open(bin_path, "wb") as bin_f, open(idx_path, "wb") as idx_f:
            offset = 0
            for i in range(num_rows):
                text = ds["train"][i]["text"]
                tokens = tokenizer.encode(text, bos=True, eos=True)
                token_ids = np.array(tokens, dtype=np.uint32)
                token_ids.tofile(bin_f)
                np.array([offset], dtype=np.int64).tofile(idx_f)
                offset += len(token_ids)
                if i % 10000 == 0:
                    print(f"[pid {os.getpid()}] {file_path}: {i + 1}/{num_rows}", flush=True)
            # Trailing offset marks the end of the last document.
            np.array([offset], dtype=np.int64).tofile(idx_f)

        print(f"[pid {os.getpid()}] done {file_path} ({num_rows} rows)", flush=True)
        with open(log_file_path, "a") as log_f:
            log_f.write(f"{datetime.now()}: done {file_path} ({num_rows} rows)\n")
        gc.collect()
    except Exception as e:  # noqa: BLE001 - log and continue so one bad file does not abort the run
        print(f"error processing {file_path}: {e}", flush=True)
        with open(log_file_path, "a") as log_f:
            log_f.write(f"{datetime.now()}: error processing {file_path}: {e}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-glob", required=True, help="Glob for input parquet files, e.g. '/data/fineweb/**/*.parquet'.")
    parser.add_argument("--output-dir", required=True, help="Directory to write .bin/.idx shards into.")
    parser.add_argument("--tokenizer-path", default="./assets/tokenizer.model", help="Path to the Llama-2 SentencePiece model.")
    parser.add_argument("--shard-prefix", default="fineweb_tokens", help="Filename prefix for the output shards.")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of worker processes.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "tokenize.log")

    files = sorted(glob.glob(args.input_glob))
    print(f"found {len(files)} parquet files for glob {args.input_glob!r}", flush=True)
    if not files:
        raise SystemExit("No parquet files matched --input-glob.")

    work = [
        (fp, i, args.output_dir, args.tokenizer_path, args.shard_prefix, log_file_path)
        for i, fp in enumerate(files)
    ]
    with Pool(processes=args.num_workers) as pool:
        pool.starmap(process_single_file, work)

    print("all files processed", flush=True)
    with open(log_file_path, "a") as log_f:
        log_f.write(f"{datetime.now()}: all files processed\n")


if __name__ == "__main__":
    main()
