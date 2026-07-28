# DATA-1: Minimal Deterministic `asr2soap` Split Plan

**Goal:** Create one Python 3.11 script that loads the Hugging Face dataset,
combines `train` and `validation`, filters `asr2soap`, transforms nested
messages into explicit columns, and deterministically writes five compact
Parquet datasets.

**Architecture:** `scripts/split_asr2soap.py` is the only production data
pipeline file. It has a loader and a splitter/transformer. The script uses
the unique source `id` for all split assignments, emits a readable
zero-padded `unique_id`, and discards unused source metadata. Generated
Parquet files are disposable and can be recreated on local, HPC, or AWS from
the same script arguments.

**Tech Stack:** Python 3.11, `venv`, Hugging Face `datasets`/
`huggingface_hub`, pyarrow, pytest, and Ruff. Conda is not used.

**Spec:** [DATA-1 Design Spec](../specs/data-1-deterministic-asr2soap-split.md)

---

## Intended Usage

The core pipeline is two steps and returns Hugging Face dataset objects:

```python
dataset = load_asr2soap(revision)
splits = split_dataset(dataset, seed=42)

sft_train = splits["sft"]
rm_train = splits["reward_model"]
grpo_train = splits["grpo"]
validation = splits["validation"]
test = splits["test"]
```

Each resulting dataset has exactly these columns:

```text
unique_id, original_dataset, prompt, transcript, soap_note, split
```

The initial run writes five Parquet files. Later SFT, reward-model, GRPO,
validation, and evaluation code loads only the relevant split file.

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Existing | `requirements.txt` | Python 3.11 dependencies for a standard `venv` |
| Modify | `.gitignore` | Ignore regenerated `data/` output and existing local environment paths |
| Create | `scripts/split_asr2soap.py` | Load, transform, split, validate, and write output files |
| Create | `tests/test_split_asr2soap.py` | Focused in-memory tests for splitter functions |

The schema review has been completed manually. The deleted notebook is not
part of the implementation. The repository owner remains responsible for all
Git staging, commits, and pushes.

---

## Task 1: Create the Python 3.11 Environment Files

**Status:** Complete

- `requirements.txt` was created.
- The existing `.gitignore` already ignored `.venv` and `.env`; `data/` was
  added for regenerated split output.
- A Python 3.11.14 `.venv` was created and dependencies installed.

---

## Task 2: Implement the Single Splitter Script

**Files:**
- Create: `scripts/split_asr2soap.py`

- [ ] **Step 1: Define the fixed source, schema, and split policy**

Keep all DATA-1 decisions visibly near the top of the one script.

```python
DATASET_REPO = "YapayNet/betrac2026-augmented"
SOURCE_SPLITS = ("train", "validation")
TASK = "asr2soap"
EXPECTED_ROW_COUNT = 18_794
SEED = 42
SPLIT_FRACTIONS = {
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
```

- [ ] **Step 2: Load, combine, and filter the public source dataset**

The script accepts a full Hugging Face commit SHA; it does not require a
token for this public dataset and does not read/write credentials.

```python
def load_asr2soap(revision: str) -> Dataset:
    source_datasets = [
        load_dataset(DATASET_REPO, split=split_name, revision=revision)
        for split_name in SOURCE_SPLITS
    ]
    combined = concatenate_datasets(source_datasets)
    return combined.filter(lambda row: row["task"] == TASK)
```

- [ ] **Step 3: Validate IDs, assign readable IDs, and split deterministically**

Sort the unique source `id` values, create the readable `unique_id` mapping,
then use a dedicated seeded random generator to assign the source IDs to the
five project splits. `source_key` is intentionally ignored.

```python
def split_dataset(dataset: Dataset, seed: int = SEED) -> DatasetDict:
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
        shuffled_ids,
        rounded_counts(len(shuffled_ids), SPLIT_FRACTIONS),
    )
    return build_derived_splits(dataset, unique_id_by_source_id, split_by_source_id)
```

`rounded_counts` uses floor values followed by the fixed order `sft`,
`reward_model`, `grpo`, `validation`, and `test` for any remaining IDs.

- [ ] **Step 4: Extract messages and drop unused columns**

Every row must have the expected three messages. Extract their `content`
values and remove every original column in the same transformation.

```python
def build_derived_splits(
    dataset: Dataset,
    unique_id_by_source_id: dict[str, str],
    split_by_source_id: dict[str, str],
) -> DatasetDict:
    def transform(row: dict) -> dict:
        messages = row["messages"]
        if len(messages) != 3 or any(
            not isinstance(message, dict) or not message.get("content")
            for message in messages
        ):
            raise ValueError(f"Invalid messages for source id {row['id']}")
        return {
            "unique_id": unique_id_by_source_id[row["id"]],
            "original_dataset": row["original_dataset"],
            "prompt": messages[0]["content"],
            "transcript": messages[1]["content"],
            "soap_note": messages[2]["content"],
            "split": split_by_source_id[row["id"]],
        }

    derived = dataset.map(transform, remove_columns=dataset.column_names)
    return DatasetDict({
        split_name: derived.filter(lambda row, name=split_name: row["split"] == name)
        for split_name in SPLIT_FRACTIONS
    })
```

This explicitly drops `split`, `task`, `input_modality`, `output_type`,
`source_key`, `source_file`, audio metadata, `webdataset_key`, `messages`,
and original `id`.

- [ ] **Step 5: Write five Parquet files and small run metadata**

```python
def write_splits(
    splits: DatasetDict,
    *,
    revision: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataset in splits.items():
        split_dataset.to_parquet(output_dir / f"{split_name}.parquet")

    metadata = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": revision,
        "seed": SEED,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "output_columns": OUTPUT_COLUMNS,
        "split_fractions": SPLIT_FRACTIONS,
        "counts": {name: len(split_dataset) for name, split_dataset in splits.items()},
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
```

- [ ] **Step 6: Add a minimal command-line entry point**

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Full Hugging Face commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_asr2soap(args.revision)
    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(dataset)}")
    write_splits(split_dataset(dataset), revision=args.revision, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
```

Expected command:

```bash
python scripts/split_asr2soap.py \
  --revision <full-hugging-face-commit-sha> \
  --output-dir /explicit/cache/path/asr2soap_split_v1
```

**Checkpoint:** Ask for confirmation before finalizing this script.

---

## Task 3: Add Focused Tests and Run the First Split

**Files:**
- Create: `tests/test_split_asr2soap.py`
- Generated outside Git: five Parquet files and `split_metadata.json`

- [ ] **Step 1: Test the critical invariants with in-memory datasets**

```python
def test_unique_id_is_sequential_and_zero_padded() -> None:
    splits = split_dataset(fixture_dataset)
    all_rows = [row for split in splits.values() for row in split]
    assert sorted(row["unique_id"] for row in all_rows) == ["00001", "00002", "00003"]

def test_repeated_source_key_does_not_reject_distinct_rows() -> None:
    fixture = Dataset.from_list([
        valid_row(id="record-a", source_key="shared"),
        valid_row(id="record-b", source_key="shared"),
    ])
    splits = split_dataset(fixture, seed=42)
    assert sum(len(split) for split in splits.values()) == 2

def test_output_schema_is_exact() -> None:
    row = next(iter(split_dataset(fixture_dataset)["sft"]))
    assert set(row) == set(OUTPUT_COLUMNS)
```

Also test rejected duplicate source IDs, rejected invalid message count,
allocation totals, and stable assignment when input rows are reordered.

- [ ] **Step 2: Run quality checks**

```bash
ruff check scripts tests
ruff format --check scripts tests
python -m pytest tests/test_split_asr2soap.py
```

- [ ] **Step 3: Run once against the pinned real revision**

```bash
python scripts/split_asr2soap.py \
  --revision <full-hugging-face-commit-sha> \
  --output-dir /explicit/cache/path/asr2soap_split_v1
```

Review the five output schemas, the consecutive `unique_id` range, and
`split_metadata.json`. The output is not committed; rerun the same command
elsewhere to recreate it.

**Checkpoint:** Ask for confirmation before treating this as the project
dataset version or moving to SFT work.

---

## Rollback / Risk Notes

- Delete a failed generated output directory and rerun; it is a cache, not
  a source of truth.
- A different Hugging Face revision, seed, expected row count, or output
  schema defines a new split version and must write to a new output
  directory.
- If message structure or source-ID uniqueness fails validation, stop rather
  than creating partially transformed data.
- Do not use `git add`, `git commit`, or `git push`; the repository owner
  handles those actions.
