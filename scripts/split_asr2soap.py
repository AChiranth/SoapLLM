"""Create deterministic project splits from the BeTraC `asr2soap` task."""

import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

DATASET_REPO = "YapayNet/betrac2026-augmented"
SOURCE_SPLITS = ("train", "validation")
TASK = "asr2soap"
EXPECTED_ROW_COUNT = 18_794
SEED = 42

SPLIT_FRACTIONS: dict[str, float] = {
    "sft": 0.60,
    "reward_model": 0.15,
    "grpo": 0.15,
    "validation": 0.05,
    "test": 0.05,
}

OUTPUT_COLUMNS = [
    "unique_id",
    "original_dataset",
    "prompt",
    "transcript",
    "soap_note",
    "split",
]


def load_asr2soap(revision: str) -> Dataset:
    """Load, combine, and filter the source text-to-SOAP records."""
    source_datasets = [
        load_dataset(DATASET_REPO, split=split_name, revision=revision)
        for split_name in SOURCE_SPLITS
    ]
    combined = concatenate_datasets(source_datasets)
    return combined.filter(lambda row: row["task"] == TASK)


def rounded_counts(total: int) -> dict[str, int]:
    """Allocate all source IDs with fixed-order integer rounding."""
    counts = {
        split_name: math.floor(total * fraction)
        for split_name, fraction in SPLIT_FRACTIONS.items()
    }
    remaining = total - sum(counts.values())
    for split_name in SPLIT_FRACTIONS:
        if remaining == 0:
            break
        counts[split_name] += 1
        remaining -= 1
    return counts


def assign_contiguous_ids(
    shuffled_ids: list[str], counts: dict[str, int]
) -> dict[str, str]:
    """Map each shuffled source ID to one of the five project splits."""
    assignments: dict[str, str] = {}
    start = 0
    for split_name in SPLIT_FRACTIONS:
        end = start + counts[split_name]
        assignments.update(
            {source_id: split_name for source_id in shuffled_ids[start:end]}
        )
        start = end

    if len(assignments) != len(shuffled_ids):
        raise ValueError("Every source id must receive exactly one split")
    return assignments


def build_split_assignments(
    dataset: Dataset, seed: int = SEED
) -> tuple[dict[str, str], dict[str, str]]:
    """Create stable readable IDs and five-way assignments from source IDs."""
    source_ids = dataset["id"]
    if any(source_id is None or not str(source_id).strip() for source_id in source_ids):
        raise ValueError("Every source id must be non-empty")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Source id must be unique")

    ordered_ids = sorted(source_ids)
    unique_id_by_source_id = {
        source_id: f"{index:05d}"
        for index, source_id in enumerate(ordered_ids, start=1)
    }
    shuffled_ids = ordered_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    split_by_source_id = assign_contiguous_ids(
        shuffled_ids, rounded_counts(len(shuffled_ids))
    )
    return unique_id_by_source_id, split_by_source_id


def build_derived_splits(
    dataset: Dataset,
    unique_id_by_source_id: dict[str, str],
    split_by_source_id: dict[str, str],
) -> DatasetDict:
    """Extract the final six-column schema and return its five split datasets."""

    def transform(row: dict) -> dict:
        messages = row["messages"]
        if len(messages) != 3 or any(
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        ):
            raise ValueError(f"Invalid messages for source id {row['id']}")

        return {
            "unique_id": unique_id_by_source_id[row["id"]],
            "original_dataset": row["original_dataset"],
            "prompt": messages[0]["content"].strip(),
            "transcript": messages[1]["content"].strip(),
            "soap_note": messages[2]["content"].strip(),
            "split": split_by_source_id[row["id"]],
        }

    derived = dataset.map(transform, remove_columns=dataset.column_names)
    derived = derived.select_columns(OUTPUT_COLUMNS)
    return DatasetDict(
        {
            split_name: derived.filter(
                lambda row, name=split_name: row["split"] == name
            )
            for split_name in SPLIT_FRACTIONS
        }
    )


def split_dataset(dataset: Dataset, seed: int = SEED) -> DatasetDict:
    """Assign source IDs, extract output fields, and return five project splits."""
    unique_id_by_source_id, split_by_source_id = build_split_assignments(
        dataset, seed=seed
    )
    return build_derived_splits(dataset, unique_id_by_source_id, split_by_source_id)


def write_splits(
    splits: DatasetDict,
    *,
    revision: str,
    output_dir: Path,
) -> None:
    """Write the five derived split files and compact reproducibility metadata."""
    if set(splits) != set(SPLIT_FRACTIONS):
        raise ValueError("Expected exactly the five configured project splits")

    try:
        output_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new split version."
        ) from error

    for split_name, split_dataset in splits.items():
        if split_dataset.column_names != OUTPUT_COLUMNS:
            raise ValueError(f"Unexpected output schema for {split_name}")
        split_dataset.to_parquet(str(output_dir / f"{split_name}.parquet"))

    metadata = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": revision,
        "seed": SEED,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "output_columns": OUTPUT_COLUMNS,
        "split_fractions": SPLIT_FRACTIONS,
        "counts": {
            split_name: len(split_dataset)
            for split_name, split_dataset in splits.items()
        },
    }
    metadata_path = output_dir / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    """Load the pinned source revision and materialize its five split files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision", required=True, help="Full Hugging Face commit SHA"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_asr2soap(args.revision)
    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(dataset)}")
    write_splits(
        split_dataset(dataset, seed=SEED),
        revision=args.revision,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
