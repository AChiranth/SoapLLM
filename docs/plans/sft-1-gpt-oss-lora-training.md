# SFT-1: GPT-OSS-20B LoRA SFT Implementation Plan

**Status:** Draft  
**Spec:** [SFT-1 GPT-OSS-20B LoRA Supervised Fine-Tuning](../specs/sft-1-gpt-oss-lora-training.md)

---

## Goal

Implement one reproducible CUDA-only CLI that reads the validated DATA-1
Parquet cache, fine-tunes the pinned GPT-OSS-20B revision with LoRA, evaluates
on the Validation split each epoch, and saves adapter/checkpoint/run metadata.
The task remains open until the repository owner completes the planned
one-epoch deterministic subset POC on HPC.

## Architecture

`scripts/train_sft.py` remains the single production SFT entry point. It uses
the model tokenizer's Harmony chat template through TRL's `SFTTrainer`, with
`assistant_only_loss=True`; current TRL supplies the GPT-OSS training template
needed to obtain assistant-token masks. The script never hand-assembles
Harmony tokens.

```text
DATA-1 Parquet cache
       │ validate manifest + load sft/validation only
       ▼
row prompt/transcript/soap_note → conversational messages
       │ tokenizer/TRL Harmony template; assistant-only loss
       ▼
pinned GPT-OSS-20B MXFP4 base → PEFT LoRA adapter → SFTTrainer
       │ eval + save each epoch
       ▼
trainer checkpoints, adapter exports, metrics, immutable run metadata
```

## Files

| Action | File | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add PyTorch and the GPU-training Python packages |
| Create | `scripts/train_sft.py` | Validation, data preparation, model/LoRA setup, training, checkpointing, metadata, and CLI |
| Create | `tests/test_train_sft.py` | Focused CPU-only tests for deterministic, non-model behavior |
| Modify | `.gitignore` | Ignore the default local SFT-artifact directory if one is introduced by the CLI |

PyTorch is a required runtime dependency. On the HPC it must be installed from
the CUDA wheel index matching the cluster driver; local development may install
a CPU/macOS wheel for imports and CPU-only tests, but never executes training.

---

## Task 1: Declare the Training Dependencies

**Files:** `requirements.txt`

Add bounded compatible dependencies for the implementation:

```text
# SFT-1 training. On HPC, satisfy torch with the CUDA wheel selected for the
# cluster driver before installing the remaining requirements.
torch>=2.5,<3.0
accelerate>=1.0,<2.0
peft>=0.17,<2.0
transformers>=4.55,<6.0
trl>=0.20,<2.0
```

Do not add an external experiment-tracking dependency, `bitsandbytes`, or an
HF token to requirements. The model uses its native MXFP4 format rather than a
bitsandbytes 4-bit loader. The eventual HPC environment must be verified to
provide a TRL release whose GPT-OSS training chat template supports
assistant-only loss; the script will fail clearly if no assistant labels are
produced.

**Validation:**

```bash
# Select the Linux/CUDA wheel from pytorch.org for the cluster, then install
# the remaining repository requirements.
python -m pip install torch --index-url <pytorch-cuda-wheel-index>
python -m pip install -r requirements.txt
python -c "import torch, accelerate, peft, transformers, trl; print(torch.cuda.is_available())"
```

The CUDA installation/verification is for the HPC environment; it is not a
requirement to run model training on the local Mac.

## Task 2: Build and Test Pure Data/Configuration Helpers First

**Files:**
- Create: `tests/test_train_sft.py`
- Create: `scripts/train_sft.py`

Start the script with constants and pure helpers so the high-value behavior is
testable without CUDA, model weights, or network access:

```python
MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
DATASET_REVISION = "cce6a4f2afc9203f491aec3a98d5c08121315dc5"
DATASET_SEED = 42
EXPECTED_COUNTS = {"sft": 11277, "validation": 939}

def to_messages(row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": row["prompt"]},
        {"role": "user", "content": row["transcript"]},
        {"role": "assistant", "content": row["soap_note"]},
    ]

def choose_sft_subset(rows, *, seed: int, max_samples: int | None,
                      fraction: float | None):
    # Reject two selection modes, sort unique_id, shuffle with Random(seed),
    # and return the requested deterministic prefix.
    ...
```

Implement helpers to:

1. load and validate `split_metadata.json`, including DATA-1 revision, seed,
   exact schema, and SFT/Validation counts;
2. load only `sft.parquet` and `validation.parquet`;
3. convert derived rows to conversational `messages` without changing text;
4. choose an integer or fractional SFT subset deterministically and calculate
   a SHA-256 digest of selected IDs;
5. validate CLI values and immutable resume fields; and
6. build the documented MoE expert parameter names from layer indices.

Add tests before GPU-specific code:

```python
def test_messages_preserve_the_three_data_fields() -> None:
    assert to_messages(fixture_row) == [
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": "dialogue"},
        {"role": "assistant", "content": "reference SOAP"},
    ]

def test_subset_is_stable_when_source_row_order_changes() -> None:
    assert selected_ids(rows) == selected_ids(list(reversed(rows)))

def test_conflicting_subset_flags_are_rejected() -> None:
    with pytest.raises(ValueError):
        choose_sft_subset(rows, seed=42, max_samples=10, fraction=0.1)
```

Tests also cover malformed manifests, invalid fractions/counts, selected-ID
hash stability, resume-metadata mismatch, and expert target-name construction.

## Task 3: Add Assistant-Only Formatting Guards

**Files:**
- Modify: `scripts/train_sft.py`
- Modify: `tests/test_train_sft.py`

Prepare a conversational `messages` column for TRL and validate every row
before training. Use the pinned tokenizer's `apply_chat_template` only to
measure formatted length; do not construct token strings directly. Reject any
row whose untruncated formatted length exceeds `--max-length` and include its
`unique_id` in the error.

```python
def validate_formatted_lengths(dataset, tokenizer, max_length: int) -> None:
    for row in dataset:
        encoded = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        if len(encoded) > max_length:
            raise ValueError(
                f"Example {row['unique_id']} has {len(encoded)} tokens; "
                f"max_length is {max_length}"
            )

training_args = SFTConfig(
    max_length=args.max_length,
    assistant_only_loss=True,
    packing=False,
    ...,
)
```

Instantiate `SFTTrainer` with raw conversational rows and the pinned
tokenizer. The script must assert after trainer preparation that assistant loss
labels exist, contain at least one trainable token for each checked batch, and
mask non-assistant tokens as `-100`. If the installed TRL/template combination
cannot produce those masks, fail before `trainer.train()` with an upgrade
message rather than falling back to full-sequence loss.

Unit-test the label assertion against a small fake batch containing
`input_ids`, `labels`, and known masked positions. Do not download GPT-OSS in
the test suite.

## Task 4: Implement CUDA, Model, and LoRA Initialization

**Files:** `scripts/train_sft.py`

Add a GPU preflight immediately before any model loading. It checks CUDA
availability, exactly one visible target device, BF16 support, and logs device
name/total memory. It must fail on the local Mac before any attempt to fetch
the 20B model.

Load the model and adapter as follows:

```python
if not torch.cuda.is_available():
    raise RuntimeError("SFT-1 requires a CUDA GPU; run training on the HPC.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    quantization_config=Mxfp4Config(dequantize=True),
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    use_cache=False,
    device_map="auto",
)
peft_model = get_peft_model(
    model,
    LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        target_parameters=expert_target_parameters(args.expert_layers),
    ),
)
```

Default to rank 8, alpha 16, dropout 0, and expert layers 7/15/23. Print and
record every trainable parameter name/count, and assert base-model parameters
are frozen. Keep the LoRA and hardware knobs explicit CLI arguments; do not
auto-tune them from detected VRAM.

## Task 5: Implement Training, Epoch Artifacts, and Resume Protection

**Files:**
- Modify: `scripts/train_sft.py`
- Modify: `.gitignore` if a default artifact path is introduced

Create one required `--output-dir` per run. Reject an existing output path
unless `--resume-from-checkpoint` is provided. Before a resume, compare stored
immutable values with the resolved invocation:

```python
IMMUTABLE_FIELDS = (
    "model_id", "model_revision", "dataset_revision", "dataset_seed",
    "selected_ids_sha256", "max_length", "lora_config", "message_format",
)
```

Build `SFTConfig` with the specified defaults: three epochs, micro-batch 1,
gradient accumulation 16, BF16, gradient checkpointing, evaluation and
checkpoint save each epoch, and `report_to="none"`. Use a deterministic seed
for the trainer and data selection.

Use a small `TrainerCallback` to export the adapter and tokenizer after each
completed epoch, while the trainer's normal checkpoint captures optimizer and
scheduler state:

```python
class SaveAdapterEachEpoch(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **_):
        epoch_dir = Path(args.output_dir) / f"adapter-epoch-{state.epoch:g}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
```

Write `resolved_config.json`, `run_metadata.json`, and `metrics.json` with
atomic replacement where practical. Metadata includes Git revision, package
versions, CUDA device details, DATA-1 manifest contents, selected-ID digest,
model revision, LoRA layout, and final metrics. Never serialize environment
variables, Hugging Face credentials, or model weights into JSON.

## Task 6: Finish Focused Tests and Static Validation

**Files:** `tests/test_train_sft.py`

Complete the test suite with mocked tokenizers/rows and temporary directories:

```python
def test_overlength_message_reports_unique_id() -> None:
    tokenizer = FakeTokenizer(length=2049)
    with pytest.raises(ValueError, match="00042"):
        validate_formatted_lengths(dataset, tokenizer, max_length=2048)

def test_resume_rejects_changed_lora_layout(tmp_path: Path) -> None:
    write_run_metadata(tmp_path, lora_config={"r": 8})
    with pytest.raises(ValueError, match="lora_config"):
        validate_resume_metadata(tmp_path, lora_config={"r": 16})
```

Run:

```bash
ruff check scripts tests
ruff format --check scripts tests
python -m pytest tests/test_split_asr2soap.py tests/test_train_sft.py
```

No test invokes CUDA, downloads the base model, accesses Hugging Face, or runs
the 20B model locally.

## Task 7: HPC Proof of Concept (Repository Owner)

After code review and the repository owner's Git commit/push, run the
one-epoch subset POC on the HPC against staged DATA-1 files. The command will
follow this shape, with real paths provided by the operator:

```bash
python scripts/train_sft.py \
  --data-dir /shared/data/asr2soap_split_v2 \
  --output-dir /shared/runs/sft-1-poc \
  --num-train-epochs 1 \
  --max-train-samples 512
```

The repository owner reports the command outcome, logs, peak memory, and
artifact paths back in this session. SFT-1 is complete only after the run
loads the pinned model/data, evaluates Validation loss, and emits a resumable
checkpoint, adapter export, metrics, and run metadata without error.

## Rollback / Risk Notes

- No implementation step mutates the DATA-1 Parquet cache or the base model.
- Failed output directories and checkpoint files are external artifacts; keep
  their logs/metadata for diagnosis, then choose a new output directory for a
  corrected run.
- An OOM is handled by reducing explicit sequence-length or micro-batch
  settings and rerunning; never silently alter LoRA rank or dataset selection.
- If the installed TRL release cannot produce GPT-OSS assistant masks, stop and
  update the explicit dependency rather than accepting loss over prompts.
- The repository owner alone performs Git staging, commits, pushes, HPC job
  submission, and any future Hugging Face upload.
