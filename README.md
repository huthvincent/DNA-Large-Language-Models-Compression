# FOCUS: Summary-Token Compression Toolkit

This folder provides a cleaned-up subset of the original `bionemo_evo2_project_AB_compress` repository, focusing on the evaluation utilities for the summary-token (FOCUS) adapter trained on Evo-2 7B. Summary tokens (sometimes called *FOCUS tokens*) replace the old “Activation Beacon” terminology: they are the compact, trainable markers inserted every *k* bases to retain long-range context.

> **Citation**  
> Zhu, R., Zhou, X., Tang, H., Scherer, S. W., & Ohno-Machado, L. (2025). *Near-Lossless Model Compression Enables Longer Context Inference in DNA Large Language Models*, pp. 2025–11.

---

## 1. Environment Setup

1. **Docker image (BioNeMo)**  
   ```bash
   export BIONEMO_IMAGE_PATH="nvcr.io/nvidia/clara/bionemo-framework:2.6.1"
   docker pull "$BIONEMO_IMAGE_PATH"    # ~8 GB
   ```
   Launch the container with GPU access, mounting this FOCUS directory and your checkpoint/output folders.

2. **Repository layout**  
   Inside the container, assume `/workspace/FOCUS` is the mount point for this folder and `/workspace/bionemo-framework` hosts the full BioNeMo stack (required for the `beacon.*` Python modules). Set:
   ```bash
   export PYTHONPATH=/workspace/bionemo-framework/workspace/vendor:\
/workspace/bionemo-framework/workspace/vendor/bionemo/sub-packages/bionemo-core/src:\
/workspace/bionemo-framework/workspace/vendor/bionemo/sub-packages/bionemo-llm/src:\
/workspace/bionemo-framework/workspace/vendor/bionemo/sub-packages/bionemo-evo2/src:\
/workspace/bionemo-framework/workspace/vendor/bionemo/3rdparty/NeMo:\
/workspace/bionemo-framework/workspace/vendor/bionemo/3rdparty/Megatron-LM:\
/workspace/FOCUS:$PYTHONPATH
   ```

3. **Model checkpoints**  
   - Base Evo-2 7B flattened weights: place under `checkpoints/evo2_7b_nemo_flattened` (or adjust paths when running scripts).  
   - Summary adapter weights: copy your `beacon_adapter.pt` into `output/focus_runs/chr1/checkpoints/`. Only the configuration (`configs/summary_config.generated.yaml`) is shipped here—the actual `.pt` file must come from your internal training run.

---

## 2. Minimal Data Samples

For reproducibility, `data/` includes short placeholder files (e.g., `data/GRCh38/chr2_sample.fasta`, `data/Virus/virus_val_sample.csv`, and an OpenGenome2 tarball). Replace them with full datasets for real experiments—the samples merely keep the project runnable in CI or documentation builds.

---

## 3. Key Scripts

All scripts live in `scripts/` and default to the structure under `output/focus_runs/chr1`. Pass `--help` to view full arguments.

### 3.1 Summary-token distribution metrics (GRCh38)

```bash
python scripts/eval_summary_metrics.py \
  --run-dir output/focus_runs/chr1 \
  --base-model checkpoints/evo2_7b_nemo_flattened \
  --chroms 2,3 \
  --segment-length 1024 \
  --num-seqs 10 \
  --sample random \
  --batch-size 4 \
  --output output/focus_runs/chr1/outputs/GRCh38/summary.tsv
```
This compares next-token distributions between the summary adapter and the frozen base model on GRCh38 chromosome slices, writing both per-chromosome CSVs and a TSV summary. Summary tokens and base tokens are evaluated with the same tokenizer; only the adapter’s internal attention differs.

### 3.2 OpenGenome2 batch evaluation

```bash
python scripts/eval_summary_metrics_opengenome2.py \
  --run-dir output/focus_runs/chr1 \
  --base-model checkpoints/evo2_7b_nemo_flattened \
  --tar-path data/OpenGenome2/batch1_sample.tar \
  --segment-length 1024 \
  --num-seqs 100 \
  --batch-size 4 \
  --output output/focus_runs/chr1/outputs/OpenGenome2/summary.tsv
```
Reads FASTA entries from a tarball (gzipped or plain) and samples random segments to assess the adapter vs. baseline divergence.

### 3.3 Virus validation evaluation

```bash
python scripts/eval_summary_metrics_virus.py \
  --run-dir output/focus_runs/chr1 \
  --base-model checkpoints/evo2_7b_nemo_flattened \
  --csv-path data/Virus/virus_val_sample.csv \
  --segment-length 1024 \
  --num-seqs 200 \
  --batch-size 4 \
  --output output/focus_runs/chr1/outputs/Virus/summary.tsv
```
Loads two-column CSVs (`sequence,label`) and samples segments uniformly at random, mirroring the GRCh38 workflow.

### 3.4 Memory profiling

```bash
python scripts/eval_memory.py \
  --run-dir output/focus_runs/chr1 \
  --base-model checkpoints/evo2_7b_nemo_flattened \
  --lengths "5000,10000,20000,40000,80000,160000" \
  --chunk-size 1024 \
  --output output/focus_runs/chr1/outputs/memory/memory.csv \
  --save-fasta
```
Measures the GPU-memory footprint for (i) the base Evo-2 forward (full-sequence) and (ii) the summary adapter’s streaming inference. It records `peak_alloc_mb` and `peak_reserved_mb`, plus the number of retained summary tokens, for downstream plotting.

---

## 4. Configuration Quick Reference

`configs/summary_config.generated.yaml` mirrors the original Activation-Beacon run with terminology updated for summary tokens. Important fields:

- `insert_every_n = 100`: place one summary token after every 100 DNA bases.
- `beacons_per_block = 1`: one summary token per block; each token summarizes exactly one $k$-mer.
- `condense_ratio = 100`: expected key/value compression factor (keep ~1% of KV entries).
- `block_len = 1024`: streaming chunk size (also used by `eval_memory.py`).
- `special_tokens.beacon_token = '~'`: the actual summary token symbol and ID (126). We keep the key name for compatibility, but treat it as the “FOCUS token” throughout the documentation.

Train/eval hyperparameters (learning rate, warm-up, dataset paths, etc.) are identical to the chr1 Activation-Beacon experiment and can be tweaked in this YAML before re-running training.

---

## 5. Step-by-Step Workflow

1. **Pull the BioNeMo container** (`docker pull "$BIONEMO_IMAGE_PATH"`).
2. **Run the container** with GPUs and mount:
   ```bash
   docker run --gpus all -it \
     -v /data2/zhu11/FOCUS:/workspace/FOCUS \
     -v /data2/zhu11/checkpoints:/workspace/checkpoints \
     -v /data2/zhu11/output:/workspace/output \
     "$BIONEMO_IMAGE_PATH" bash
   ```
3. **Set `PYTHONPATH`** to include the BioNeMo vendor subpackages (see Section 1.2).
4. **Copy checkpoints**: put `beacon_adapter.pt` into `output/focus_runs/chr1/checkpoints/` and the Evo-2 weights into `checkpoints/evo2_7b_nemo_flattened/`.
5. **(Optional) Replace sample data** under `data/` with the full GRCh38/OpenGenome2/Virus datasets.
6. **Run evaluations** using the commands in Section 3 (GRCh38, OpenGenome2, Virus) or profile memory via Section 3.4.
7. **Inspect outputs** in `output/focus_runs/chr1/outputs/*`—each script writes TSV summaries plus per-dataset CSVs.

This completes the streamlined FOCUS project setup. The scripts, minimal data, and configuration files here are enough to document or regression-test the summary-token adapter; swap in your full datasets and trained checkpoints for production-scale experiments.
