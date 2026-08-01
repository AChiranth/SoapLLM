"""Focused CPU-only tests for SFT-1 data and configuration helpers."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_sft import (
    DATASET_REVISION,
    DATASET_SEED,
    EXPECTED_COUNTS,
    EXPERT_PARAMETER_SUFFIXES,
    IMMUTABLE_FIELDS,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_COLUMNS,
    assert_assistant_only_labels,
    assert_trainer_assistant_only_loss,
    atomic_write_json,
    attach_lora_adapter,
    build_argument_parser,
    build_save_adapter_each_epoch_callback,
    build_sft_trainer,
    choose_sft_subset,
    cuda_preflight,
    expert_target_parameters,
    initialize_run_artifacts,
    load_and_validate_split_metadata,
    load_pinned_gpt_oss_model_and_tokenizer,
    load_sft_and_validation_datasets,
    prepare_conversation_dataset,
    prompt_consistency_report,
    select_sft_dataset,
    selected_ids_sha256,
    to_messages,
    trainable_parameter_summary,
    validate_cli_values,
    validate_formatted_lengths,
    validate_resume_metadata,
    write_trainer_metrics,
)


def valid_metadata() -> dict:
    """Return the DATA-1 fields SFT-1 requires from a split manifest."""
    return {
        "dataset_revision": DATASET_REVISION,
        "seed": DATASET_SEED,
        "output_columns": OUTPUT_COLUMNS,
        "counts": {
            "sft": EXPECTED_COUNTS["sft"],
            "validation": EXPECTED_COUNTS["validation"],
        },
    }


def write_metadata(tmp_path: Path, metadata: object) -> Path:
    """Write a fixture split manifest and return its containing directory."""
    (tmp_path / "split_metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def write_split_files(data_dir: Path) -> None:
    """Create empty file fixtures for the two DATA-1 files SFT may read."""
    for split_name in ("sft", "validation"):
        (data_dir / f"{split_name}.parquet").touch()


def test_messages_preserve_the_three_data_fields() -> None:
    row = {
        "prompt": "instruction with trailing spaces  ",
        "transcript": "dialogue\nwith an internal line break",
        "soap_note": "reference SOAP\n",
    }

    assert to_messages(row) == [
        {"role": "system", "content": "instruction with trailing spaces  "},
        {"role": "user", "content": "dialogue\nwith an internal line break"},
        {"role": "assistant", "content": "reference SOAP\n"},
    ]


class FakeDataset:
    """Small test double for the Dataset.map interface used by SFT-1."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def map(self, function: object) -> "FakeDataset":
        return FakeDataset([{**row, **function(row)} for row in self.rows])  # type: ignore[operator]


class FakeTokenizer:
    """Return a configured sequence length through the chat-template API."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.calls.append((messages, tokenize, add_generation_prompt))
        return list(range(self.length))


def test_prepare_conversation_dataset_adds_row_messages() -> None:
    dataset = FakeDataset(
        [
            {
                "unique_id": "00001",
                "prompt": "instruction",
                "transcript": "dialogue",
                "soap_note": "reference SOAP",
            }
        ]
    )

    prepared = prepare_conversation_dataset(dataset)

    assert prepared.rows[0]["messages"] == [
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": "dialogue"},
        {"role": "assistant", "content": "reference SOAP"},
    ]


def test_overlength_message_reports_unique_id() -> None:
    tokenizer = FakeTokenizer(length=2049)
    dataset = [
        {
            "unique_id": "00042",
            "messages": [{"role": "user", "content": "dialogue"}],
        }
    ]

    with pytest.raises(ValueError, match="00042"):
        validate_formatted_lengths(dataset, tokenizer, max_length=2048)
    assert tokenizer.calls == [
        (dataset[0]["messages"], True, False),
    ]


def test_formatted_length_at_limit_is_accepted() -> None:
    validate_formatted_lengths(
        [
            {
                "unique_id": "00042",
                "messages": [{"role": "user", "content": "dialogue"}],
            }
        ],
        FakeTokenizer(length=2048),
        max_length=2048,
    )


def test_assistant_only_labels_accept_valid_masking() -> None:
    assert_assistant_only_labels(
        labels=[[-100, -100, 11, 12], [-100, 21, -100]],
        assistant_masks=[[0, 0, 1, 1], [0, 1, 0]],
    )


@pytest.mark.parametrize(
    ("labels", "assistant_masks", "match"),
    [
        ([[-100, 11]], [[0, 0]], "No assistant tokens"),
        ([[-100, -100]], [[0, 1]], "Assistant token is masked"),
        ([[13, 11]], [[0, 1]], "Non-assistant token is trainable"),
    ],
)
def test_assistant_only_labels_reject_invalid_masking(
    labels: list[list[int]], assistant_masks: list[list[int]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        assert_assistant_only_labels(labels, assistant_masks)


class FakeSFTConfig:
    """Capture SFTConfig keyword arguments without importing TRL."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeSFTTrainer:
    """Prepare a one-example assistant-only batch for trainer guard tests."""

    def __init__(
        self,
        *,
        model: object,
        args: FakeSFTConfig,
        train_dataset: FakeDataset,
        eval_dataset: FakeDataset,
        processing_class: FakeTokenizer,
        callbacks: list[object] | None = None,
    ) -> None:
        del model, eval_dataset, processing_class, callbacks
        self.args = args
        self.train_dataset = [
            {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 1]}
            for _ in train_dataset.rows
        ]

    @staticmethod
    def data_collator(examples: list[dict]) -> dict:
        return {
            "labels": [
                [
                    token if is_assistant else -100
                    for token, is_assistant in zip(
                        example["input_ids"], example["assistant_masks"]
                    )
                ]
                for example in examples
            ]
        }


def test_build_sft_trainer_enables_assistant_only_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = FakeDataset(
        [
            {
                "unique_id": "00001",
                "prompt": "instruction",
                "transcript": "dialogue",
                "soap_note": "reference SOAP",
            }
        ]
    )
    tokenizer = FakeTokenizer(length=3)
    monkeypatch.setattr(
        "train_sft._load_trl_components", lambda: (FakeSFTConfig, FakeSFTTrainer)
    )

    trainer = build_sft_trainer(
        model=object(),
        tokenizer=tokenizer,
        train_dataset=dataset,
        validation_dataset=dataset,
        output_dir=tmp_path / "run",
        num_train_epochs=3,
        seed=42,
        max_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
    )

    assert isinstance(trainer, FakeSFTTrainer)
    assert trainer.args.kwargs["assistant_only_loss"] is True
    assert trainer.args.kwargs["packing"] is False
    assert trainer.args.kwargs["eval_strategy"] == "epoch"
    assert trainer.args.kwargs["save_strategy"] == "epoch"
    assert trainer.args.kwargs["load_best_model_at_end"] is True
    assert trainer.args.kwargs["metric_for_best_model"] == "eval_loss"
    assert trainer.args.kwargs["greater_is_better"] is False


def test_trainer_assistant_loss_guard_rejects_missing_assistant_masks() -> None:
    class TrainerWithoutAssistantMasks:
        def __init__(self) -> None:
            self.train_dataset = [{"input_ids": [1, 2, 3]}]

        @staticmethod
        def data_collator(examples: list[dict]) -> dict:
            return {"labels": [example["input_ids"] for example in examples]}

    with pytest.raises(ValueError, match="did not produce assistant masks"):
        assert_trainer_assistant_only_loss(TrainerWithoutAssistantMasks())


class FakeCuda:
    """Minimal CUDA interface for preflight tests without a GPU."""

    def __init__(
        self, *, available: bool = True, device_count: int = 1, bf16: bool = True
    ) -> None:
        self.available = available
        self.count = device_count
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    @staticmethod
    def current_device() -> int:
        return 0

    def is_bf16_supported(self) -> bool:
        return self.bf16

    @staticmethod
    def get_device_properties(device_index: int) -> SimpleNamespace:
        assert device_index == 0
        return SimpleNamespace(total_memory=48 * 1024**3)

    @staticmethod
    def get_device_name(device_index: int) -> str:
        assert device_index == 0
        return "fixture-gpu"

    @staticmethod
    def mem_get_info(device_index: int) -> tuple[int, int]:
        assert device_index == 0
        return 40 * 1024**3, 48 * 1024**3


class FakeTorch:
    """Minimal torch module supporting SFT-1 CUDA and dtype checks."""

    bfloat16 = "bf16"

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def test_cuda_preflight_reports_single_bf16_gpu() -> None:
    report = cuda_preflight(FakeTorch(FakeCuda()))

    assert report == {
        "device_index": 0,
        "device_name": "fixture-gpu",
        "total_memory_bytes": 48 * 1024**3,
        "free_memory_bytes": 40 * 1024**3,
        "available_memory_bytes": 48 * 1024**3,
    }


@pytest.mark.parametrize(
    ("cuda", "match"),
    [
        (FakeCuda(available=False), "requires a CUDA GPU"),
        (FakeCuda(device_count=2), "exactly one visible"),
        (FakeCuda(bf16=False), "BF16"),
    ],
)
def test_cuda_preflight_rejects_unsupported_hardware(
    cuda: FakeCuda, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        cuda_preflight(FakeTorch(cuda))


def test_load_pinned_gpt_oss_model_and_tokenizer_uses_required_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    class FakeMxfp4Config:
        def __init__(self, **kwargs: object) -> None:
            calls["mxfp4"] = kwargs

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> str:
            calls["tokenizer"] = (args, kwargs)
            return "tokenizer"

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def __init__(self) -> None:
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            calls["model"] = (args, kwargs)
            return FakeModel()

    monkeypatch.setattr(
        "train_sft._load_transformers_components",
        lambda: (FakeAutoTokenizer, FakeAutoModelForCausalLM, FakeMxfp4Config),
    )

    model, tokenizer, report = load_pinned_gpt_oss_model_and_tokenizer(
        FakeTorch(FakeCuda())
    )

    assert tokenizer == "tokenizer"
    assert calls["tokenizer"] == ((MODEL_ID,), {"revision": MODEL_REVISION})
    assert calls["mxfp4"] == {"dequantize": True}
    assert calls["model"] == (
        (MODEL_ID,),
        {
            "revision": MODEL_REVISION,
            "quantization_config": ANY,
            "torch_dtype": "bf16",
            "attn_implementation": "eager",
            "use_cache": False,
            "device_map": "auto",
        },
    )
    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is True
    assert report["free_memory_bytes"] == 40 * 1024**3


class FakeParameter:
    """Represent one named parameter and its trainability in LoRA tests."""

    def __init__(self, count: int, requires_grad: bool) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


def test_attach_lora_adapter_targets_all_linear_and_selected_moe_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakePeftModel:
        def __init__(self, expert_targets: list[str]) -> None:
            self.parameters = [
                ("base_model.model.embed_tokens.weight", FakeParameter(100, False)),
                (
                    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    FakeParameter(8, True),
                ),
            ] + [
                (f"base_model.{target}.lora_A.weight", FakeParameter(4, True))
                for target in expert_targets
            ]

        def named_parameters(self):
            return self.parameters

    def fake_get_peft_model(model: object, config: FakeLoraConfig) -> FakePeftModel:
        del model
        captured["config"] = config.kwargs
        return FakePeftModel(config.kwargs["target_parameters"])

    monkeypatch.setattr(
        "train_sft._load_peft_components", lambda: (FakeLoraConfig, fake_get_peft_model)
    )

    peft_model, summary = attach_lora_adapter(
        object(), lora_r=8, lora_alpha=16, lora_dropout=0.0, expert_layers=[7, 15]
    )

    assert isinstance(peft_model, FakePeftModel)
    assert captured["config"] == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "target_parameters": [
            "model.layers.7.mlp.experts.gate_up_proj",
            "model.layers.7.mlp.experts.down_proj",
            "model.layers.15.mlp.experts.gate_up_proj",
            "model.layers.15.mlp.experts.down_proj",
        ],
        "task_type": "CAUSAL_LM",
    }
    assert summary["trainable_parameters"] == 24
    assert len(summary["trainable_parameter_names"]) == 5


def test_trainable_parameter_summary_rejects_unfrozen_base_weights() -> None:
    class ModelWithUnfrozenBaseWeight:
        @staticmethod
        def named_parameters():
            return [
                ("base_model.model.layers.0.weight", FakeParameter(10, True)),
                ("base_model.model.layers.0.lora_A.weight", FakeParameter(2, True)),
            ]

    with pytest.raises(ValueError, match="Base-model parameters"):
        trainable_parameter_summary(
            ModelWithUnfrozenBaseWeight(), ["model.layers.0.mlp.experts.gate_up_proj"]
        )


def test_atomic_write_json_replaces_existing_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"old": true}\n')

    atomic_write_json(artifact_path, {"new": [1, 2]})

    assert json.loads(artifact_path.read_text()) == {"new": [1, 2]}


def test_initialize_run_artifacts_writes_new_run_records(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    metadata = valid_immutable_metadata()

    initialize_run_artifacts(
        output_dir=output_dir,
        resume_from_checkpoint=None,
        resolved_config={"max_length": 2048},
        run_metadata=metadata,
    )

    assert json.loads((output_dir / "resolved_config.json").read_text()) == {
        "max_length": 2048
    }
    assert json.loads((output_dir / "run_metadata.json").read_text()) == metadata


def test_initialize_run_artifacts_rejects_existing_new_run_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        initialize_run_artifacts(
            output_dir=output_dir,
            resume_from_checkpoint=None,
            resolved_config={},
            run_metadata=valid_immutable_metadata(),
        )


def test_initialize_run_artifacts_validates_metadata_before_resume(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run_metadata.json").write_text(
        json.dumps(valid_immutable_metadata())
    )
    checkpoint_dir = output_dir / "checkpoints" / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True)
    changed_metadata = valid_immutable_metadata()
    changed_metadata["max_length"] = 4096

    with pytest.raises(ValueError, match="max_length"):
        initialize_run_artifacts(
            output_dir=output_dir,
            resume_from_checkpoint=checkpoint_dir,
            resolved_config={},
            run_metadata=changed_metadata,
        )


def test_save_adapter_callback_exports_model_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTrainerCallback:
        pass

    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def save_pretrained(self, path: Path, **kwargs: object) -> None:
            self.calls.append((path, kwargs))
            path.mkdir()

    class FakeTokenizerForSave:
        def __init__(self) -> None:
            self.calls = []

        def save_pretrained(self, path: Path) -> None:
            self.calls.append(path)

    monkeypatch.setattr("train_sft._load_trainer_callback", lambda: FakeTrainerCallback)
    tokenizer = FakeTokenizerForSave()
    callback = build_save_adapter_each_epoch_callback(tmp_path, tokenizer)
    model = FakeModel()
    control = object()

    assert (
        callback.on_epoch_end(
            args=object(),
            state=SimpleNamespace(epoch=1.0),
            control=control,
            model=model,
        )
        is control
    )
    assert model.calls == [(tmp_path / "adapter-epoch-1", {"safe_serialization": True})]
    assert tokenizer.calls == [tmp_path / "adapter-epoch-1"]

    assert (
        callback.on_evaluate(
            args=object(),
            state=SimpleNamespace(log_history=[{"epoch": 1.0, "eval_loss": 0.5}]),
            control=control,
        )
        is control
    )
    assert json.loads((tmp_path / "metrics.json").read_text()) == [
        {"epoch": 1.0, "eval_loss": 0.5}
    ]


def test_write_trainer_metrics_persists_log_history(tmp_path: Path) -> None:
    trainer = SimpleNamespace(state=SimpleNamespace(log_history=[{"loss": 0.5}]))

    write_trainer_metrics(tmp_path, trainer)

    assert json.loads((tmp_path / "metrics.json").read_text()) == [{"loss": 0.5}]


def test_argument_parser_uses_documented_sft_defaults() -> None:
    args = build_argument_parser().parse_args(
        ["--data-dir", "data", "--output-dir", "run"]
    )

    assert args.num_train_epochs == 3
    assert args.max_length == 2048
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 16
    assert args.expert_layers == [7, 15, 23]


def test_select_sft_dataset_uses_deterministic_selected_ids() -> None:
    class FakeArrowDataset:
        def __init__(self, unique_ids: list[str]) -> None:
            self.unique_ids = unique_ids

        def __getitem__(self, column_name: str) -> list[str]:
            assert column_name == "unique_id"
            return self.unique_ids

        def select(self, indices: list[int]) -> list[str]:
            return [self.unique_ids[index] for index in indices]

    dataset = FakeArrowDataset(["00003", "00001", "00002"])

    selected_dataset, selected_ids = select_sft_dataset(
        dataset, seed=42, max_samples=2, fraction=None
    )

    assert selected_dataset == selected_ids
    assert len(selected_ids) == 2


def sft_rows() -> list[dict]:
    """Return an intentionally unordered fixture with stable DATA-1 IDs."""
    return [
        {"unique_id": unique_id}
        for unique_id in ("00004", "00001", "00006", "00002", "00005", "00003")
    ]


def valid_cli_values() -> dict:
    """Return the documented valid baseline SFT configuration."""
    return {
        "num_train_epochs": 3,
        "max_train_samples": None,
        "train_sample_fraction": None,
        "seed": 42,
        "max_length": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "expert_layers": [7, 15, 23],
    }


def valid_immutable_metadata() -> dict:
    """Return the fields that must not change when a run resumes."""
    return {
        "model_id": "openai/gpt-oss-20b",
        "model_revision": "model-revision",
        "dataset_revision": DATASET_REVISION,
        "dataset_seed": DATASET_SEED,
        "selected_ids_sha256": "selected-ids-digest",
        "sft_prompt_consistency": {
            "distinct_prompt_count": 1,
            "distinct_prompt_sha256": ["prompt-digest"],
        },
        "max_length": 2048,
        "lora_config": {"r": 8, "alpha": 16, "dropout": 0.0},
        "message_format": "system-user-assistant",
    }


def test_choose_sft_subset_is_stable_when_source_row_order_changes() -> None:
    selected = choose_sft_subset(sft_rows(), seed=42, max_samples=3, fraction=None)
    reordered = choose_sft_subset(
        list(reversed(sft_rows())), seed=42, max_samples=3, fraction=None
    )

    assert [row["unique_id"] for row in selected] == [
        row["unique_id"] for row in reordered
    ]
    assert selected_ids_sha256(selected) == selected_ids_sha256(reordered)


def test_prompt_consistency_report_hashes_unique_prompts_without_rewriting_them() -> (
    None
):
    prompt_with_spaces = "clinical instruction  "
    report = prompt_consistency_report(
        ["other instruction", prompt_with_spaces, prompt_with_spaces]
    )

    assert report == {
        "distinct_prompt_count": 2,
        "distinct_prompt_sha256": sorted(
            [
                hashlib.sha256(b"other instruction").hexdigest(),
                hashlib.sha256(prompt_with_spaces.encode("utf-8")).hexdigest(),
            ]
        ),
    }


def test_prompt_consistency_report_rejects_non_string_prompts() -> None:
    with pytest.raises(TypeError, match="prompt"):
        prompt_consistency_report(["instruction", 42])  # type: ignore[list-item]


def test_choose_sft_subset_selects_integer_and_fractional_prefixes() -> None:
    integer_subset = choose_sft_subset(
        sft_rows(), seed=42, max_samples=2, fraction=None
    )
    fractional_subset = choose_sft_subset(
        sft_rows(), seed=42, max_samples=None, fraction=0.5
    )

    assert len(integer_subset) == 2
    assert len(fractional_subset) == 3


def test_choose_sft_subset_rejects_conflicting_selection_modes() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        choose_sft_subset(sft_rows(), seed=42, max_samples=2, fraction=0.5)


@pytest.mark.parametrize(
    ("max_samples", "fraction"),
    [(0, None), (None, 0.0), (None, 1.1)],
)
def test_choose_sft_subset_rejects_invalid_selection_values(
    max_samples: int | None, fraction: float | None
) -> None:
    with pytest.raises(ValueError):
        choose_sft_subset(
            sft_rows(), seed=42, max_samples=max_samples, fraction=fraction
        )


def test_validate_cli_values_accepts_documented_baseline() -> None:
    validate_cli_values(**valid_cli_values())


def test_expert_target_parameters_builds_documented_gpt_oss_names() -> None:
    assert expert_target_parameters([7, 15, 23]) == [
        "model.layers.7.mlp.experts.gate_up_proj",
        "model.layers.7.mlp.experts.down_proj",
        "model.layers.15.mlp.experts.gate_up_proj",
        "model.layers.15.mlp.experts.down_proj",
        "model.layers.23.mlp.experts.gate_up_proj",
        "model.layers.23.mlp.experts.down_proj",
    ]
    assert EXPERT_PARAMETER_SUFFIXES == (
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    )


@pytest.mark.parametrize("expert_layers", [[], [-1], [7, 7]])
def test_expert_target_parameters_rejects_invalid_layer_indices(
    expert_layers: list[int],
) -> None:
    with pytest.raises(ValueError, match="expert_layers"):
        expert_target_parameters(expert_layers)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("num_train_epochs", 0),
        ("max_length", 0),
        ("lora_dropout", 1.0),
        ("expert_layers", [7, 7]),
    ],
)
def test_validate_cli_values_rejects_invalid_configuration(
    field: str, invalid_value: object
) -> None:
    values = valid_cli_values()
    values[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        validate_cli_values(**values)


def test_validate_resume_metadata_accepts_matching_immutable_fields(
    tmp_path: Path,
) -> None:
    metadata = valid_immutable_metadata()
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))

    validate_resume_metadata(tmp_path, metadata)


def test_validate_resume_metadata_rejects_changed_lora_layout(
    tmp_path: Path,
) -> None:
    stored_metadata = valid_immutable_metadata()
    (tmp_path / "run_metadata.json").write_text(json.dumps(stored_metadata))
    resolved_metadata = valid_immutable_metadata()
    resolved_metadata["lora_config"] = {"r": 16, "alpha": 32, "dropout": 0.0}

    with pytest.raises(ValueError, match="lora_config"):
        validate_resume_metadata(tmp_path, resolved_metadata)


def test_validate_resume_metadata_requires_every_immutable_field(
    tmp_path: Path,
) -> None:
    stored_metadata = valid_immutable_metadata()
    stored_metadata.pop(IMMUTABLE_FIELDS[0])
    (tmp_path / "run_metadata.json").write_text(json.dumps(stored_metadata))

    with pytest.raises(ValueError, match=IMMUTABLE_FIELDS[0]):
        validate_resume_metadata(tmp_path, valid_immutable_metadata())


def test_load_and_validate_split_metadata_accepts_data_1_manifest(
    tmp_path: Path,
) -> None:
    data_dir = write_metadata(tmp_path, valid_metadata())

    assert load_and_validate_split_metadata(data_dir) == valid_metadata()


def test_load_and_validate_split_metadata_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="split_metadata.json"):
        load_and_validate_split_metadata(tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("dataset_revision", "different-revision"),
        ("seed", 0),
        ("output_columns", list(reversed(OUTPUT_COLUMNS))),
    ],
)
def test_load_and_validate_split_metadata_rejects_invalid_contract_field(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    metadata = valid_metadata()
    metadata[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        load_and_validate_split_metadata(write_metadata(tmp_path, metadata))


@pytest.mark.parametrize(
    ("split_name", "invalid_count"),
    [("sft", 11_276), ("validation", 940)],
)
def test_load_and_validate_split_metadata_rejects_invalid_split_count(
    tmp_path: Path, split_name: str, invalid_count: int
) -> None:
    metadata = valid_metadata()
    metadata["counts"][split_name] = invalid_count

    with pytest.raises(ValueError, match=split_name):
        load_and_validate_split_metadata(write_metadata(tmp_path, metadata))


def test_load_and_validate_split_metadata_rejects_non_object_counts(
    tmp_path: Path,
) -> None:
    metadata = valid_metadata()
    metadata["counts"] = []

    with pytest.raises(TypeError, match="counts"):
        load_and_validate_split_metadata(write_metadata(tmp_path, metadata))


def test_load_sft_and_validation_datasets_loads_only_required_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = write_metadata(tmp_path, valid_metadata())
    write_split_files(data_dir)
    calls = []
    expected_datasets = {"sft": object(), "validation": object()}

    def fake_load_parquet_datasets(data_files: dict[str, str]) -> dict:
        calls.append(data_files)
        return expected_datasets

    monkeypatch.setattr("train_sft._load_parquet_datasets", fake_load_parquet_datasets)

    assert load_sft_and_validation_datasets(data_dir) is expected_datasets
    assert calls == [
        {
            "sft": str(data_dir / "sft.parquet"),
            "validation": str(data_dir / "validation.parquet"),
        }
    ]


@pytest.mark.parametrize("missing_split", ["sft", "validation"])
def test_load_sft_and_validation_datasets_rejects_missing_required_file(
    tmp_path: Path, missing_split: str
) -> None:
    data_dir = write_metadata(tmp_path, valid_metadata())
    write_split_files(data_dir)
    (data_dir / f"{missing_split}.parquet").unlink()

    with pytest.raises(FileNotFoundError, match=f"{missing_split}.parquet"):
        load_sft_and_validation_datasets(data_dir)
