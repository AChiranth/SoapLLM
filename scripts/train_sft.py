"""Train a GPT-OSS LoRA adapter on the validated DATA-1 SFT split."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
DATASET_REVISION = "cce6a4f2afc9203f491aec3a98d5c08121315dc5"
DATASET_SEED = 42
EXPECTED_COUNTS = {"sft": 11_277, "validation": 939}
OUTPUT_COLUMNS = [
    "unique_id",
    "original_dataset",
    "prompt",
    "transcript",
    "soap_note",
    "split",
]
SFT_DATA_FILES = ("sft", "validation")
EXPERT_PARAMETER_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
IMMUTABLE_FIELDS = (
    "model_id",
    "model_revision",
    "dataset_revision",
    "dataset_seed",
    "selected_ids_sha256",
    "sft_prompt_consistency",
    "max_length",
    "lora_config",
    "message_format",
)


def atomic_write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Atomically replace a JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(payload, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_run_directory(
    output_dir: Path, resume_from_checkpoint: Path | None
) -> None:
    """Create a new run directory or validate an existing resumable run."""
    if resume_from_checkpoint is None:
        try:
            output_dir.mkdir(parents=True)
        except FileExistsError as error:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Choose a new directory or supply --resume-from-checkpoint."
            ) from error
        return

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Run directory is missing for resume: {output_dir}")
    if not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Trainer checkpoint is missing for resume: {resume_from_checkpoint}"
        )


def initialize_run_artifacts(
    *,
    output_dir: Path,
    resume_from_checkpoint: Path | None,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    """Create immutable run records or verify them before a resume."""
    prepare_run_directory(output_dir, resume_from_checkpoint)
    if resume_from_checkpoint is not None:
        validate_resume_metadata(output_dir, run_metadata)
        return

    atomic_write_json(output_dir / "resolved_config.json", resolved_config)
    atomic_write_json(output_dir / "run_metadata.json", run_metadata)


def _load_trainer_callback() -> Any:
    """Import the Transformers callback base class at training runtime."""
    from transformers import TrainerCallback

    return TrainerCallback


def build_save_adapter_each_epoch_callback(output_dir: Path, tokenizer: Any) -> Any:
    """Build a callback that writes an adapter-only export after each epoch."""
    trainer_callback = _load_trainer_callback()

    class SaveAdapterEachEpoch(trainer_callback):
        def on_epoch_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any | None = None,
            processing_class: Any | None = None,
            **_: Any,
        ) -> Any:
            del args
            if model is None:
                raise ValueError("Trainer callback did not receive the PEFT model")
            if state.epoch is None:
                raise ValueError("Trainer callback did not receive a completed epoch")

            adapter_dir = output_dir / f"adapter-epoch-{state.epoch:g}"
            if adapter_dir.exists():
                return control
            model.save_pretrained(adapter_dir, safe_serialization=True)
            (processing_class or tokenizer).save_pretrained(adapter_dir)
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            **_: Any,
        ) -> Any:
            """Persist metrics after each epoch-level Validation evaluation."""
            del args
            atomic_write_json(output_dir / "metrics.json", state.log_history)
            return control

    return SaveAdapterEachEpoch()


def write_trainer_metrics(output_dir: Path, trainer: Any) -> None:
    """Persist the Trainer's epoch-level metric history as a JSON artifact."""
    atomic_write_json(output_dir / "metrics.json", trainer.state.log_history)


def to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one derived DATA-1 row to the GPT-OSS conversation contract."""
    return [
        {"role": "system", "content": row["prompt"]},
        {"role": "user", "content": row["transcript"]},
        {"role": "assistant", "content": row["soap_note"]},
    ]


def prepare_conversation_dataset(dataset: Any) -> Any:
    """Add the unmodified DATA-1 conversation contract for TRL formatting."""
    return dataset.map(lambda row: {"messages": to_messages(row)})


def validate_formatted_lengths(
    dataset: Iterable[Mapping[str, Any]], tokenizer: Any, max_length: int
) -> None:
    """Reject a formatted conversation that would require truncation."""
    _validate_positive_integer("max_length", max_length)
    for row in dataset:
        token_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        token_count = len(token_ids)
        if token_count > max_length:
            raise ValueError(
                f"Example {row['unique_id']} has {token_count} tokens; "
                f"max_length is {max_length}"
            )


def assert_assistant_only_labels(
    labels: Iterable[Iterable[int]], assistant_masks: Iterable[Iterable[int]]
) -> None:
    """Ensure labels train only assistant tokens and include one per example."""
    label_rows = list(labels)
    mask_rows = list(assistant_masks)
    if len(label_rows) != len(mask_rows):
        raise ValueError("labels and assistant_masks must have the same batch size")

    for row_index, (label_row, mask_row) in enumerate(zip(label_rows, mask_rows)):
        labels_for_row = list(label_row)
        mask_for_row = list(mask_row)
        if len(labels_for_row) != len(mask_for_row):
            raise ValueError(
                f"labels and assistant_masks differ in length for batch row {row_index}"
            )
        if not any(mask_for_row):
            raise ValueError(f"No assistant tokens found for batch row {row_index}")

        for label, is_assistant in zip(labels_for_row, mask_for_row):
            if is_assistant and label == -100:
                raise ValueError(f"Assistant token is masked for batch row {row_index}")
            if not is_assistant and label != -100:
                raise ValueError(
                    f"Non-assistant token is trainable for batch row {row_index}"
                )


def _load_torch() -> Any:
    """Import PyTorch only when CUDA training setup begins."""
    import torch

    return torch


def cuda_preflight(torch_module: Any | None = None) -> dict[str, Any]:
    """Verify the single CUDA GPU and BF16 prerequisites for SFT-1."""
    torch = torch_module or _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("SFT-1 requires a CUDA GPU; run training on the HPC.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("SFT-1 requires exactly one visible CUDA GPU")
    device_index = torch.cuda.current_device()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT-1 requires CUDA BF16 support")

    properties = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes": free_bytes,
        "available_memory_bytes": total_bytes,
    }


def _load_transformers_components() -> tuple[Any, Any, Any]:
    """Import the pinned-model loading components at training runtime."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    return AutoTokenizer, AutoModelForCausalLM, Mxfp4Config


def load_pinned_gpt_oss_model_and_tokenizer(
    torch_module: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Preflight CUDA and load the pinned GPT-OSS tokenizer and base model."""
    torch = torch_module or _load_torch()
    preflight = cuda_preflight(torch)
    auto_tokenizer, auto_model_for_causal_lm, mxfp4_config = (
        _load_transformers_components()
    )
    tokenizer = auto_tokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = auto_model_for_causal_lm.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=mxfp4_config(dequantize=True),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        use_cache=False,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer, preflight


def _load_peft_components() -> tuple[Any, Any]:
    """Import PEFT only when LoRA adapters are being attached."""
    from peft import LoraConfig, get_peft_model

    return LoraConfig, get_peft_model


def trainable_parameter_summary(
    model: Any, expert_targets: list[str]
) -> dict[str, Any]:
    """Verify frozen base weights and describe every trainable adapter tensor."""
    named_parameters = list(model.named_parameters())
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("LoRA initialization produced no trainable parameters")

    unfrozen_base_parameters = [
        name for name, parameter in trainable if "lora_" not in name
    ]
    if unfrozen_base_parameters:
        raise ValueError(
            "Base-model parameters must remain frozen; found "
            + ", ".join(unfrozen_base_parameters)
        )

    trainable_names = [name for name, _ in trainable]
    missing_expert_targets = [
        target
        for target in expert_targets
        if not any(target in trainable_name for trainable_name in trainable_names)
    ]
    if missing_expert_targets:
        raise ValueError(
            "LoRA initialization did not adapt configured MoE expert parameters: "
            + ", ".join(missing_expert_targets)
        )

    return {
        "total_parameters": sum(parameter.numel() for _, parameter in named_parameters),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_names": trainable_names,
    }


def attach_lora_adapter(
    model: Any,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    expert_layers: list[int],
    target_modules: str = "all-linear",
) -> tuple[Any, dict[str, Any]]:
    """Attach the documented GPT-OSS LoRA adapter and verify its layout."""
    validate_lora_configuration(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        expert_layers=expert_layers,
    )
    if not isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules must be a non-empty string")
    expert_targets = expert_target_parameters(expert_layers)
    lora_config_class, get_peft_model = _load_peft_components()
    peft_model = get_peft_model(
        model,
        lora_config_class(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            target_parameters=expert_targets,
            task_type="CAUSAL_LM",
        ),
    )
    return peft_model, trainable_parameter_summary(peft_model, expert_targets)


def _load_trl_components() -> tuple[Any, Any]:
    """Import GPU-training dependencies only when trainer construction begins."""
    from trl import SFTConfig, SFTTrainer

    return SFTConfig, SFTTrainer


def assert_trainer_assistant_only_loss(trainer: Any) -> None:
    """Check a prepared SFTTrainer batch for valid assistant-only labels."""
    dataset = trainer.train_dataset
    if len(dataset) == 0:
        raise ValueError(
            "Cannot validate assistant-only loss with an empty training dataset"
        )

    examples = [dataset[index] for index in range(min(len(dataset), 2))]
    try:
        assistant_masks = [example["assistant_masks"] for example in examples]
    except KeyError as error:
        raise ValueError(
            "The installed TRL/template combination did not produce assistant masks; "
            "upgrade TRL or provide a GPT-OSS training chat template."
        ) from error

    batch = trainer.data_collator(examples)
    if "labels" not in batch:
        raise ValueError("SFTTrainer data collator did not produce labels")
    labels = batch["labels"]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    normalized_masks = []
    for label_row, mask_row in zip(labels, assistant_masks):
        mask = list(mask_row)
        normalized_masks.append(mask + [0] * (len(label_row) - len(mask)))
    assert_assistant_only_labels(labels, normalized_masks)


def build_sft_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    output_dir: Path,
    num_train_epochs: float,
    seed: int,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    callbacks: list[Any] | None = None,
) -> Any:
    """Create a guarded assistant-only TRL trainer for GPT-OSS SFT."""
    if (
        isinstance(num_train_epochs, bool)
        or not isinstance(num_train_epochs, (float, int))
        or num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs must be greater than 0")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_positive_integer("max_length", max_length)
    _validate_positive_integer(
        "per_device_train_batch_size", per_device_train_batch_size
    )
    _validate_positive_integer(
        "gradient_accumulation_steps", gradient_accumulation_steps
    )
    prepared_train_dataset = prepare_conversation_dataset(train_dataset)
    prepared_validation_dataset = prepare_conversation_dataset(validation_dataset)
    validate_formatted_lengths(prepared_train_dataset, tokenizer, max_length)
    validate_formatted_lengths(prepared_validation_dataset, tokenizer, max_length)

    sft_config_class, sft_trainer_class = _load_trl_components()
    training_args = sft_config_class(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        max_length=max_length,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = sft_trainer_class(
        model=model,
        args=training_args,
        train_dataset=prepared_train_dataset,
        eval_dataset=prepared_validation_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    assert_trainer_assistant_only_loss(trainer)
    return trainer


def _validate_subset_selection(max_samples: int | None, fraction: float | None) -> None:
    """Validate the mutually exclusive deterministic subset options."""
    if max_samples is not None and fraction is not None:
        raise ValueError("max_samples and fraction cannot both be supplied")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int)
    ):
        raise ValueError("max_samples must be a positive integer")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    if fraction is not None and (
        isinstance(fraction, bool)
        or not isinstance(fraction, (float, int))
        or not 0 < fraction <= 1
    ):
        raise ValueError("fraction must be greater than 0 and at most 1")


def _validate_positive_integer(name: str, value: Any) -> None:
    """Reject booleans, non-integers, and non-positive numeric values."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def expert_target_parameters(expert_layers: list[int]) -> list[str]:
    """Build exact GPT-OSS MoE expert parameter names for PEFT LoRA."""
    if not expert_layers:
        raise ValueError("expert_layers must contain at least one layer index")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in expert_layers
    ):
        raise ValueError("expert_layers must contain non-negative integer indices")
    if len(set(expert_layers)) != len(expert_layers):
        raise ValueError("expert_layers must not contain duplicate indices")

    return [
        f"model.layers.{layer}.{suffix}"
        for layer in expert_layers
        for suffix in EXPERT_PARAMETER_SUFFIXES
    ]


def validate_lora_configuration(
    *, lora_r: int, lora_alpha: float, lora_dropout: float, expert_layers: list[int]
) -> None:
    """Validate the LoRA and selected GPT-OSS MoE expert configuration."""
    _validate_positive_integer("lora_r", lora_r)
    if (
        isinstance(lora_alpha, bool)
        or not isinstance(lora_alpha, (float, int))
        or lora_alpha <= 0
    ):
        raise ValueError("lora_alpha must be greater than 0")
    if (
        isinstance(lora_dropout, bool)
        or not isinstance(lora_dropout, (float, int))
        or not 0 <= lora_dropout < 1
    ):
        raise ValueError("lora_dropout must be at least 0 and less than 1")
    expert_target_parameters(expert_layers)


def validate_cli_values(
    *,
    num_train_epochs: float,
    max_train_samples: int | None,
    train_sample_fraction: float | None,
    seed: int,
    max_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    expert_layers: list[int],
) -> None:
    """Validate the documented SFT CLI/configuration values before training."""
    if (
        isinstance(num_train_epochs, bool)
        or not isinstance(num_train_epochs, (float, int))
        or num_train_epochs <= 0
    ):
        raise ValueError("num_train_epochs must be greater than 0")
    _validate_subset_selection(max_train_samples, train_sample_fraction)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_positive_integer("max_length", max_length)
    _validate_positive_integer(
        "per_device_train_batch_size", per_device_train_batch_size
    )
    _validate_positive_integer(
        "gradient_accumulation_steps", gradient_accumulation_steps
    )
    validate_lora_configuration(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        expert_layers=expert_layers,
    )


def choose_sft_subset(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples: int | None,
    fraction: float | None,
) -> list[dict[str, Any]]:
    """Select a deterministic SFT subset from rows with unique IDs."""
    _validate_subset_selection(max_samples, fraction)

    sorted_rows = sorted(rows, key=lambda row: row["unique_id"])
    unique_ids = [row["unique_id"] for row in sorted_rows]
    if any(not isinstance(unique_id, str) or not unique_id for unique_id in unique_ids):
        raise ValueError("Every selected row must have a non-empty string unique_id")
    if len(set(unique_ids)) != len(unique_ids):
        raise ValueError("SFT rows must have unique unique_id values")

    random.Random(seed).shuffle(sorted_rows)
    if max_samples is not None:
        return sorted_rows[:max_samples]
    if fraction is not None:
        return sorted_rows[: math.floor(len(sorted_rows) * fraction)]
    return sorted_rows


def selected_ids_sha256(rows: list[dict[str, Any]]) -> str:
    """Return a stable digest identifying the selected DATA-1 rows."""
    selected_ids = sorted(row["unique_id"] for row in rows)
    return hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()


def prompt_consistency_report(prompts: Iterable[str]) -> dict[str, Any]:
    """Return an observation-only, order-independent report for SFT prompts."""
    prompt_values = list(prompts)
    if any(not isinstance(prompt, str) for prompt in prompt_values):
        raise TypeError("Every SFT prompt must be a string")

    distinct_prompt_hashes = sorted(
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in set(prompt_values)
    )
    return {
        "distinct_prompt_count": len(distinct_prompt_hashes),
        "distinct_prompt_sha256": distinct_prompt_hashes,
    }


def select_sft_dataset(
    dataset: Any,
    *,
    seed: int,
    max_samples: int | None,
    fraction: float | None,
) -> tuple[Any, list[str]]:
    """Select a deterministic Arrow subset while materializing only unique IDs."""
    rows = [{"unique_id": unique_id} for unique_id in dataset["unique_id"]]
    selected_rows = choose_sft_subset(
        rows, seed=seed, max_samples=max_samples, fraction=fraction
    )
    selected_ids = [row["unique_id"] for row in selected_rows]
    index_by_unique_id = {
        unique_id: index for index, unique_id in enumerate(dataset["unique_id"])
    }
    return dataset.select(
        [index_by_unique_id[unique_id] for unique_id in selected_ids]
    ), selected_ids


def _package_versions() -> dict[str, str]:
    """Return installed training-package versions without reading environment values."""
    versions = {}
    for package_name in ("torch", "accelerate", "peft", "transformers", "trl"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def _git_revision() -> str:
    """Return the current Git revision without failing a training run outside Git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_run_metadata(
    *,
    split_metadata: Mapping[str, Any],
    selected_ids: list[str],
    sft_prompt_consistency: Mapping[str, Any],
    max_length: int,
    lora_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build credential-free, reproducible SFT-1 metadata for one run."""
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_seed": DATASET_SEED,
        "selected_ids_count": len(selected_ids),
        "selected_ids_sha256": selected_ids_sha256(
            [{"unique_id": unique_id} for unique_id in selected_ids]
        ),
        "sft_prompt_consistency": dict(sft_prompt_consistency),
        "max_length": max_length,
        "lora_config": dict(lora_config),
        "message_format": "system-user-assistant",
        "dataset_manifest": dict(split_metadata),
        "git_revision": _git_revision(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the SFT-1 command-line interface with documented defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--train-sample-fraction", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--expert-layers", type=int, nargs="+", default=[7, 15, 23])
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def run_sft(args: argparse.Namespace) -> None:
    """Execute one guarded SFT run or resume a compatible interrupted run."""
    validate_cli_values(
        num_train_epochs=args.num_train_epochs,
        max_train_samples=args.max_train_samples,
        train_sample_fraction=args.train_sample_fraction,
        seed=args.seed,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        expert_layers=args.expert_layers,
    )
    split_metadata = load_and_validate_split_metadata(args.data_dir)
    datasets = load_sft_and_validation_datasets(args.data_dir)
    sft_prompt_consistency = prompt_consistency_report(datasets["sft"]["prompt"])
    print(
        f"SFT prompt consistency: {json.dumps(sft_prompt_consistency, sort_keys=True)}"
    )
    selected_train_dataset, selected_ids = select_sft_dataset(
        datasets["sft"],
        seed=args.seed,
        max_samples=args.max_train_samples,
        fraction=args.train_sample_fraction,
    )
    lora_config = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "expert_layers": args.expert_layers,
        "expert_target_parameters": expert_target_parameters(args.expert_layers),
    }
    resolved_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_metadata = build_run_metadata(
        split_metadata=split_metadata,
        selected_ids=selected_ids,
        sft_prompt_consistency=sft_prompt_consistency,
        max_length=args.max_length,
        lora_config=lora_config,
    )
    initialize_run_artifacts(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
    )

    model, tokenizer, cuda_report = load_pinned_gpt_oss_model_and_tokenizer()
    peft_model, parameter_summary = attach_lora_adapter(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        expert_layers=args.expert_layers,
        target_modules=args.target_modules,
    )
    print(f"CUDA preflight: {json.dumps(cuda_report, sort_keys=True)}")
    print(
        "Trainable parameters: "
        f"{parameter_summary['trainable_parameters']}/"
        f"{parameter_summary['total_parameters']}"
    )
    for parameter_name in parameter_summary["trainable_parameter_names"]:
        print(f"Trainable adapter parameter: {parameter_name}")
    if args.resume_from_checkpoint is None:
        run_metadata["cuda_preflight"] = cuda_report
        run_metadata["trainable_parameter_summary"] = parameter_summary
        atomic_write_json(args.output_dir / "run_metadata.json", run_metadata)

    trainer = build_sft_trainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=selected_train_dataset,
        validation_dataset=datasets["validation"],
        output_dir=args.output_dir / "checkpoints",
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        callbacks=[build_save_adapter_each_epoch_callback(args.output_dir, tokenizer)],
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    write_trainer_metrics(args.output_dir, trainer)


def main() -> None:
    """Run the SFT-1 CLI."""
    run_sft(build_argument_parser().parse_args())


def validate_resume_metadata(
    run_dir: Path, resolved_metadata: Mapping[str, Any]
) -> None:
    """Reject resumption when a run's immutable inputs have changed."""
    metadata_path = run_dir / "run_metadata.json"
    try:
        stored_metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Run metadata is missing for resume: {metadata_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Run metadata is not valid JSON: {metadata_path}") from error

    if not isinstance(stored_metadata, dict):
        raise TypeError("Run metadata must contain a JSON object")
    for field in IMMUTABLE_FIELDS:
        if field not in stored_metadata:
            raise ValueError(
                f"Stored run metadata is missing immutable field {field!r}"
            )
        if field not in resolved_metadata:
            raise ValueError(f"Resolved metadata is missing immutable field {field!r}")
        if stored_metadata[field] != resolved_metadata[field]:
            raise ValueError(f"Cannot resume: immutable field {field!r} differs")


def load_and_validate_split_metadata(data_dir: Path) -> dict[str, Any]:
    """Load and validate the immutable DATA-1 split manifest."""
    metadata_path = data_dir / "split_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"DATA-1 split metadata is missing: {metadata_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"DATA-1 split metadata is not valid JSON: {metadata_path}"
        ) from error

    if not isinstance(metadata, dict):
        raise TypeError("DATA-1 split metadata must contain a JSON object")

    expected_fields = {
        "dataset_revision": DATASET_REVISION,
        "seed": DATASET_SEED,
        "output_columns": OUTPUT_COLUMNS,
    }
    for field, expected_value in expected_fields.items():
        actual_value = metadata.get(field)
        if actual_value != expected_value:
            raise ValueError(
                f"DATA-1 split metadata field {field!r} must be "
                f"{expected_value!r}; found {actual_value!r}"
            )

    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        raise TypeError("DATA-1 split metadata field 'counts' must be an object")
    for split_name, expected_count in EXPECTED_COUNTS.items():
        actual_count = counts.get(split_name)
        if actual_count != expected_count:
            raise ValueError(
                f"DATA-1 split metadata count for {split_name!r} must be "
                f"{expected_count}; found {actual_count!r}"
            )

    return metadata


def _load_parquet_datasets(data_files: dict[str, str]) -> Any:
    """Load Parquet files through Hugging Face Datasets at training runtime."""
    from datasets import load_dataset

    return load_dataset("parquet", data_files=data_files)


def load_sft_and_validation_datasets(data_dir: Path) -> Any:
    """Load only the SFT and Validation Parquet files from a DATA-1 cache."""
    load_and_validate_split_metadata(data_dir)
    data_files = {
        split_name: str(data_dir / f"{split_name}.parquet")
        for split_name in SFT_DATA_FILES
    }
    missing_files = [path for path in data_files.values() if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "DATA-1 split file is missing: "
            + ", ".join(str(path) for path in missing_files)
        )

    datasets = _load_parquet_datasets(data_files)
    if set(datasets) != set(SFT_DATA_FILES):
        raise ValueError("Expected only SFT and Validation datasets")
    return datasets


if __name__ == "__main__":
    main()
