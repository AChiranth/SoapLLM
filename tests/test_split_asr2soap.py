"""Focused tests for deterministic ASR-to-SOAP dataset splitting."""

import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split_asr2soap import OUTPUT_COLUMNS, rounded_counts, split_dataset, write_splits


def valid_row(record_id: str, source_key: str = "shared") -> dict:
    """Build a minimal source row with the expected message layout."""
    return {
        "id": record_id,
        "original_dataset": "betrac-fixture",
        "source_key": source_key,
        "messages": [
            {"content": f"prompt for {record_id}"},
            {"content": f"transcript for {record_id}"},
            {"content": f"SOAP note for {record_id}"},
        ],
    }


@pytest.fixture
def fixture_dataset() -> Dataset:
    return Dataset.from_list(
        [valid_row(f"record-{index:02d}") for index in range(1, 21)]
    )


def all_rows(splits: dict[str, Dataset]) -> list[dict]:
    return [row for split in splits.values() for row in split]


def test_unique_id_is_sequential_and_zero_padded(fixture_dataset: Dataset) -> None:
    rows = all_rows(split_dataset(fixture_dataset))

    assert sorted(row["unique_id"] for row in rows) == [
        f"{index:05d}" for index in range(1, 21)
    ]


def test_repeated_source_key_does_not_reject_distinct_rows() -> None:
    fixture = Dataset.from_list(
        [valid_row("record-a", "shared"), valid_row("record-b", "shared")]
    )

    splits = split_dataset(fixture, seed=42)

    assert sum(len(split) for split in splits.values()) == 2


def test_output_schema_is_exact(fixture_dataset: Dataset) -> None:
    rows = all_rows(split_dataset(fixture_dataset))

    assert rows
    assert all(list(row) == OUTPUT_COLUMNS for row in rows)


def test_extracted_text_is_trimmed_without_changing_internal_formatting() -> None:
    row = valid_row("record-a")
    row["messages"] = [
        {"content": "  prompt  "},
        {"content": "\n transcript\nwith internal line break \t"},
        {"content": "\t SOAP   note \n"},
    ]

    extracted_row = all_rows(split_dataset(Dataset.from_list([row])))[0]

    assert extracted_row["prompt"] == "prompt"
    assert extracted_row["transcript"] == "transcript\nwith internal line break"
    assert extracted_row["soap_note"] == "SOAP   note"


def test_duplicate_source_ids_are_rejected() -> None:
    fixture = Dataset.from_list([valid_row("duplicate"), valid_row("duplicate")])

    with pytest.raises(ValueError, match="Source id must be unique"):
        split_dataset(fixture)


def test_invalid_message_count_is_rejected() -> None:
    row = valid_row("record-a")
    row["messages"] = row["messages"][:2]

    with pytest.raises(ValueError, match="Invalid messages for source id record-a"):
        split_dataset(Dataset.from_list([row]))


def test_allocation_totals_match_input_size() -> None:
    for total in (0, 1, 3, 20, 18_794):
        assert sum(rounded_counts(total).values()) == total


def test_assignment_is_stable_when_input_rows_are_reordered(
    fixture_dataset: Dataset,
) -> None:
    reordered = fixture_dataset.select(list(reversed(range(len(fixture_dataset)))))

    original_assignments = {
        row["unique_id"]: row["split"]
        for row in all_rows(split_dataset(fixture_dataset))
    }
    reordered_assignments = {
        row["unique_id"]: row["split"] for row in all_rows(split_dataset(reordered))
    }

    assert reordered_assignments == original_assignments


def test_existing_output_directory_is_rejected(
    fixture_dataset: Dataset, tmp_path: Path
) -> None:
    output_dir = tmp_path / "existing-split"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Choose a new split version"):
        write_splits(
            split_dataset(fixture_dataset),
            revision="test-revision",
            output_dir=output_dir,
        )


def test_new_output_directory_receives_all_split_files(
    fixture_dataset: Dataset, tmp_path: Path
) -> None:
    output_dir = tmp_path / "new-split"
    splits = split_dataset(fixture_dataset)

    write_splits(splits, revision="test-revision", output_dir=output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "grpo.parquet",
        "reward_model.parquet",
        "sft.parquet",
        "split_metadata.json",
        "test.parquet",
        "validation.parquet",
    ]
    metadata = json.loads((output_dir / "split_metadata.json").read_text())
    assert metadata["counts"] == {
        split_name: len(split) for split_name, split in splits.items()
    }
