# Data preparation

This directory holds the **offline** data tooling — scripts you run once, by
hand, before training. They turn raw data into the tokenized `.bin` / `.idx`
shards that the trainer reads.

At **train time**, the data is consumed by library code under
`torchtitan/datasets/` (you do not call these directly):

| Stage | Code | When it runs |
| --- | --- | --- |
| Download the tokenizer | `scripts/data/download_tokenizer.py` | once, offline |
| Tokenize parquet → bin/idx | `scripts/data/tokenize_fineweb.py` | once, offline |
| Load bin/idx during training | `torchtitan/datasets/hf_datasets.py` (`BinIdxDataset`) | every run, automatically |
| Tokenizer implementation | `torchtitan/datasets/tokenizer/` (`create_tokenizer`) | every run, automatically |

## 0. Tokenizer

The Llama-2 SentencePiece tokenizer is already included at
`assets/tokenizer.model`. To fetch it yourself instead:

```bash
python scripts/data/download_tokenizer.py \
    --repo_id meta-llama/Llama-2-7b \
    --hf_token <your-hf-token> \
    --local_dir assets/
```

## 1. Source data

We pretrain on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
(a specific Common Crawl snapshot, e.g. `CC-MAIN-2023-50`), downloaded as parquet
files from the HuggingFace Hub. Any text dataset exposed as parquet with a
`"text"` column works the same way.

## 2. Tokenization: parquet → `.bin` / `.idx`

`tokenize_fineweb.py` streams each parquet file and tokenizes it with the
Llama-2 SentencePiece tokenizer, adding BOS/EOS per document. For every input
parquet file it writes one shard pair:

- **`<prefix>_shard_<id>.bin`** — all document token ids concatenated, stored as
  little-endian `uint32` (4 bytes per token).
- **`<prefix>_shard_<id>.idx`** — `int64` document-start offsets into the `.bin`
  file, with a trailing offset equal to the total token count. Document `i`
  therefore spans tokens `[idx[i], idx[i+1])`.

This is exactly the binary layout that `BinIdxDataset` memory-maps at train time.

```bash
python scripts/data/tokenize_fineweb.py \
    --input-glob "/path/to/fineweb/data/CC-MAIN-2023-50/*.parquet" \
    --output-dir /path/to/fineweb_tokenized \
    --tokenizer-path ./assets/tokenizer.model \
    --shard-prefix fineweb_tokens \
    --num-workers 32
```

## 3. Pointing the trainer at the data

The training configs take the tokenized directories directly:

```toml
[training]
dataset = "fineweb"
dataset_train_path = "/path/to/fineweb_tokenized/train"
dataset_val_path   = "/path/to/fineweb_tokenized/val"
```

`build_hf_data_loader` detects a directory containing `.bin` / `.idx` files and
uses `BinIdxDataset` automatically; the `dataset` name is just a label in that
case. How you partition the tokenized shards into `train/` and `val/` is up to
you (e.g. hold out a few shards for validation) and is not prescribed here.
