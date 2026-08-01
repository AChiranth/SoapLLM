# SFT-1: GPT-OSS-20B LoRA Supervised Fine-Tuning

**Date:** 2026-07-31  
**Status:** Draft  
**Parent Plan:** [Post-Training Clinical Scribe Project Plan](../Post_Training_Clinical_Scribe_Project_Plan_v2.md)  
**Dataset Contract:** [DATA-1 Deterministic `asr2soap` Split](data-1-deterministic-asr2soap-split.md)

---

## Problem Statement

The project has a reproducible, encounter-disjoint `asr2soap` dataset but
does not yet have training code to adapt GPT-OSS-20B from a doctor--patient
dialogue to a structured clinical SOAP note. The implementation must load an
immutable base-model revision, preserve the dataset's per-row instruction,
update only LoRA adapter parameters, and save resumable checkpoints and
reproducibility metadata.

## Goals

- Provide a Python 3.11 SFT entry point for one CUDA GPU using Hugging Face
  Transformers, PEFT, and TRL.
- Pin the base model to `openai/gpt-oss-20b` revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`, resolved on 2026-07-31.
- Load only `sft.parquet` for optimization and only `validation.parquet` for
  evaluation; never load Reward Model, GRPO, or Test data into this stage.
- Train a LoRA adapter over the model's native MXFP4 checkpoint, leaving all
  base-model weights frozen.
- Preserve each row's existing `prompt` as the instruction, rather than
  replacing it with a project-global prompt.
- Default to three epochs over all 11,277 SFT rows, while supporting a
  reproducible one-epoch/subset proof of concept.
- Save an adapter-only checkpoint at the end of every epoch and support
  resuming from a saved trainer checkpoint.
- Record the resolved configuration, model and dataset revisions, selected
  data IDs, trainable-parameter count, environment, metrics, and Git revision
  alongside each run.

## Non-Goals

- Do not perform full-parameter fine-tuning, reward-model training, GRPO,
  preference generation, GPT-5 judging, or Test-split evaluation.
- Do not implement validation-note generation, SOAP-structure scoring, or a
  dry-run mode in this task.
- Do not upload a model, adapter, dataset, or run artifacts to Hugging Face.
- Do not support AWS Lambda as a training target. Lambda is not a GPU training
  service; the script targets an HPC or cloud GPU environment.
- Do not modify the DATA-1 split algorithm or regenerate the derived corpus.
- Do not fabricate reasoning traces or clinical facts not present in the
  reference SOAP note.

## Relevant Codepaths

| Path | Role |
|---|---|
| `docs/Post_Training_Clinical_Scribe_Project_Plan_v2.md` | Governing SFT objective, data split, SOAP structure, and proof-of-concept gate |
| `docs/specs/data-1-deterministic-asr2soap-split.md` | Input Parquet schema and immutable dataset revision |
| `scripts/split_asr2soap.py` | Produces `sft.parquet`, `validation.parquet`, and `split_metadata.json` |
| `requirements.txt` | Existing Python 3.11 preprocessing dependencies |
| `scripts/train_sft.py` | New single-GPU SFT CLI entry point |
| `tests/test_train_sft.py` | New focused CPU-only tests for data formatting and configuration behavior |

## Input and Conversation Contract

The run requires a pre-materialized DATA-1 output directory on node-local or
shared storage accessible to the training job. SFT does not download the
source dataset or recreate the split at runtime. The directory contains:

```text
data/asr2soap_split_v2/
├── sft.parquet
├── validation.parquet
└── split_metadata.json
```

Before training, the script must verify the metadata's dataset revision,
seed, output schema, and expected SFT/Validation counts. A mismatched or
missing manifest fails before the model is loaded.

The script reads `sft.parquet` and `validation.parquet` through Hugging Face
Datasets/Arrow. This permits memory-mapped, on-demand access where the
platform supports it; the entire corpus is not treated as a prerequisite
in-memory Python object. Tokenization and batch collation may use bounded host
memory, but the canonical source remains the pre-materialized Parquet cache.

For each row, construct exactly this logical conversation:

```python
[
    {"role": "system", "content": row["prompt"]},
    {"role": "user", "content": row["transcript"]},
    {"role": "assistant", "content": row["soap_note"]},
]
```

GPT-OSS's tokenizer chat template renders a custom `system` instruction in
its developer-instruction channel. The script must therefore call the pinned
tokenizer's `apply_chat_template`; it must not concatenate prose or special
tokens itself. The model's assistant target contains the reference SOAP note
only. No synthetic analysis/reasoning field is introduced.

Training labels must mask the instruction, transcript, chat-template control
tokens, and padding with `-100`. Cross-entropy loss is computed only for the
assistant SOAP-note completion. If a formatted example exceeds the configured
maximum length, the script must fail with the offending `unique_id` by default
rather than silently truncate clinical content. A future, explicitly approved
policy may add deterministic truncation.

At startup, report the number of distinct prompt values and stable hashes of
those values. This is an observation-only consistency check; it must not
deduplicate or rewrite rows.

## Model and LoRA Design

Use `AutoTokenizer` and `AutoModelForCausalLM` with the model ID and exact
revision above. The implementation follows OpenAI's GPT-OSS Transformers
guidance: native MXFP4 loading with `Mxfp4Config(dequantize=True)`, BF16
compute, `use_cache=False`, gradient checkpointing, and eager attention during
training. The model is a mixture-of-experts architecture, so the PEFT adapter
must cover both standard linear layers and configured expert projection
parameters.

The default LoRA baseline is deliberately small and hardware-independent:

```text
rank: 8
alpha: 16
dropout: 0.0
target_modules: all-linear
expert target layers: 7, 15, 23
expert target parameters per selected layer: gate_up_proj, down_proj
```

Rank, alpha, dropout, selected expert layers, and target modules are explicit
CLI/config values. Raising rank changes the number of trainable adapter
parameters, but it is not the primary GPU-scaling mechanism. GPU profile
changes should first tune maximum sequence length, per-device batch size, and
gradient accumulation while preserving the effective batch size and recording
the exact values in the run metadata.

The initial 48-GB profile is a starting configuration, not a guarantee of
fit: it uses a micro-batch of 1, maximum length 2,048, gradient accumulation
of 16, BF16, and gradient checkpointing. CUDA memory availability is checked
and logged before model loading. Larger-memory profiles may raise micro-batch
size or sequence length; all profiles remain configurable from the CLI.

## Training, Checkpoints, and Validation

The CLI defaults to three epochs over the entire SFT split. It exposes:

```text
--num-train-epochs             default: 3
--max-train-samples            default: unset (all rows)
--train-sample-fraction        default: unset (all rows)
--seed                         default: 42
--max-length                   default: 2048
--per-device-train-batch-size  default: 1
--gradient-accumulation-steps  default: 16
--lora-r / --lora-alpha / --lora-dropout
--output-dir                   required run-artifact directory
--resume-from-checkpoint       optional
```

`--max-train-samples` and `--train-sample-fraction` are mutually exclusive.
When either is supplied, select rows deterministically by sorting `unique_id`,
shuffling with the supplied seed, and taking the requested prefix. Persist the
selected IDs' count and SHA-256 digest in run metadata. A proof of concept can
therefore use one epoch and a small stable subset, while an invocation without
those flags is the three-epoch, full-SFT default.

The trainer evaluates loss on the untouched Validation split at each epoch,
saves a resumable trainer checkpoint plus an adapter-only export at each
epoch, and writes final train/evaluation metrics. Checkpoint selection for the
initial run is lowest validation loss unless a later approved generation-based
selector supersedes it.

The project plan's post-epoch validation-note generation and SOAP-structure
gate are intentionally deferred from SFT-1. The one-epoch subset proof of
concept is an HPC execution of this completed training script; its operational
review and any generation evaluator belong to a follow-on task.

## Reproducibility and Artifacts

Each `--output-dir` must be new unless `--resume-from-checkpoint` is supplied.
It contains at least:

```text
run_metadata.json       # resolved inputs, IDs digest, seed, Git/environment data
resolved_config.json    # all effective CLI/model/LoRA/training values
checkpoints/            # resumable Trainer state, one per epoch
adapter-epoch-*/        # PEFT adapter config and weights, one per epoch
metrics.json            # training and validation metrics by epoch
```

The script must never write credentials to these artifacts. The output path is
outside Git by default (or explicitly ignored); the repository owner decides
which lightweight metadata records to commit. Model weights, checkpoints, and
generated datasets remain external artifacts.

## Dependencies

Add the GPU training dependencies with compatible lower bounds from the
official GPT-OSS Transformers example: `transformers>=4.55.0`,
`peft>=0.17.0`, and `trl>=0.20.0`, plus `accelerate`. PyTorch is installed for
the target CUDA runtime using the appropriate PyTorch index, rather than
assuming that a macOS development environment can execute CUDA training.
The implementation must not add a dependency on an external logging service.

## Risks and Open Questions

| Risk / question | Handling |
|---|---|
| 48-GB GPU may not fit the initial MXFP4-dequantized training configuration | Start with the POC profile, log peak memory, then lower length/accumulation or use a larger GPU; never silently change the experiment configuration. |
| Long transcripts or notes exceed 2,048 tokens | Fail with IDs and token-length summary first; choose a documented truncation or larger-context policy only after review. |
| Training node cannot access the derived Parquet cache | Stage the validated DATA-1 directory on node-local scratch or a mounted shared filesystem before job submission; do not re-download/re-split within SFT. |
| Prompt strings unexpectedly differ | Report count and hashes without changing them; inspect before treating prompt uniformity as an assumption. |
| Adapter targeting for GPT-OSS MoE is incomplete | Use the documented linear and expert projection targets; print and persist trainable parameter names/count for review. |
| CLI resume accidentally changes a run | Compare the resolved immutable fields (model/dataset revision, data subset, formatting, and LoRA layout) with stored metadata and fail on a mismatch. |
| Full three-epoch run bypasses the POC gate | The default supports three epochs, but the recommended first HPC invocation is a one-epoch subset/full-data POC followed by explicit operator review. |

## Acceptance Criteria

- [ ] `train_sft.py` rejects non-CUDA execution before attempting to load the
      20B model and prints a useful GPU-memory preflight report.
- [ ] The exact GPT-OSS model ID and revision are used for both tokenizer and
      model loading and are recorded in every run.
- [ ] The script validates DATA-1 metadata and loads only SFT/Validation
      Parquet files for this stage.
- [ ] Each row uses its own `prompt`, `transcript`, and `soap_note` in the
      tokenizer's GPT-OSS chat template.
- [ ] Loss labels cover only assistant SOAP-note tokens; padding and prompt
      tokens cannot contribute to loss.
- [ ] Base-model parameters remain frozen and the persisted PEFT adapter has
      nonzero, reported trainable parameters, including configured MoE experts.
- [ ] A no-flag run defaults to three epochs and every SFT row; one epoch plus
      a deterministic subset is available through documented CLI flags.
- [ ] Validation loss, checkpoint, adapter export, metrics, and immutable run
      metadata are written once per completed epoch.
- [ ] Resuming a compatible interrupted run continues from its checkpoint;
      incompatible run metadata is rejected.
- [ ] Focused unit tests run without downloading GPT-OSS or requiring a GPU,
      and Ruff passes on new Python files.
- [ ] The repository owner runs a one-epoch deterministic subset POC on the
      HPC. It must load the pinned model and staged DATA-1 files, complete
      training and Validation-loss evaluation, and produce a resumable
      checkpoint, adapter export, metrics, and run metadata without error.

## Validation

`tests/test_train_sft.py` will cover, without downloading GPT-OSS or requiring
a GPU:

- DATA-1 manifest/schema validation and rejection of invalid inputs;
- conversion of a fixture row to the three-message conversation contract;
- assistant-only label masking using a small test tokenizer;
- deterministic integer and fractional SFT subset selection;
- rejection of conflicting subset flags and invalid CLI/configuration values;
- immutable run-metadata comparison for safe resume; and
- construction of the documented LoRA expert target names.

Run the focused checks before HPC submission:

```bash
ruff check scripts tests
ruff format --check scripts tests
python -m pytest tests/test_train_sft.py
```

The one-epoch subset proof of concept is run on the HPC after these code tests
pass. The repository owner will push the implementation, execute that run, and
report its result back in this session. SFT-1 remains open until this end-to-end
HPC POC succeeds. Job submission and generated-note quality evaluation remain
outside the code changes in SFT-1; the parent project's later SOAP-structure
quality gate is not waived by this engineering POC.

## Rollback

No source model or DATA-1 file is mutated. A failed run is recoverable by
preserving its metadata/logs, selecting a new output directory, adjusting the
explicit hardware configuration, and rerunning. Adapter and trainer
checkpoints are disposable external artifacts; the pinned model revision and
DATA-1 metadata allow the run to be reconstructed.

## References

- [OpenAI: Fine-tuning GPT-OSS with Hugging Face Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers)
- [Hugging Face model metadata for `openai/gpt-oss-20b`](https://huggingface.co/api/models/openai/gpt-oss-20b)
