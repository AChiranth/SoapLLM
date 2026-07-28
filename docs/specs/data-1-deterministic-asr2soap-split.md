# DATA-1: Deterministic `asr2soap` Data Split

**Date:** 2026-07-28  
**Status:** Draft — revised after schema and whitespace review  
**Parent Plan:** [Post-Training Clinical Scribe Project Plan](../Post_Training_Clinical_Scribe_Project_Plan_v2.md)

---

## Problem Statement

Before training begins, the project needs one reproducible five-way split of
the `asr2soap` rows in `YapayNet/betrac2026-augmented`. The source contains
metadata and a nested three-message structure that are unnecessary in the
derived training files. The splitter must produce compact, explicit columns
without mixing the five training/evaluation splits.

## Goals

- Provide one small Python script that loads, filters, transforms, and splits
  the dataset.
- Merge source `train` and `validation`, then retain only
  `task == "asr2soap"`.
- Use the unique source `id` as the split unit. Do not group by `source_key`:
  identical `source_key` values may still represent separate dialogues and
  SOAP summaries.
- Assign deterministic zero-padded `unique_id` strings from `"00001"` to
  `"18794"` for the current filtered corpus.
- Use seed `42` to assign records to SFT 60%, Reward Model 15%, GRPO 15%,
  Validation 5%, and Test 5%.
- Replace `messages` with explicit `prompt`, `transcript`, and `soap_note`
  columns, trimming only leading and trailing whitespace from each extracted
  text value, and write the assigned split to `split`.
- Record source revision, seed, retained schema, and split counts in a small
  metadata JSON next to generated files.

## Non-Goals

- Do not retain a notebook; the schema review is complete.
- Do not download audio shards, process audio, train a model, call GPT-5, or
  perform external annotation.
- Do not build a data framework, package hierarchy, manifest service, or
  external artifact store.
- Do not commit downloaded data or generated split files to Git.

## Relevant Files

| Path | Role |
|---|---|
| `docs/Post_Training_Clinical_Scribe_Project_Plan_v2.md` | Governing project and data-contract decisions |
| `docs/specs/data-1-deterministic-asr2soap-split.md` | This specification |
| `docs/plans/data-1-deterministic-asr2soap-split.md` | Implementation plan |
| `scripts/split_asr2soap.py` | Proposed single production splitter script |
| `tests/test_split_asr2soap.py` | Focused in-memory splitter tests |

## Derived Output Schema

Every output row contains only:

```python
{
    "unique_id": "00001",  # string, zero-padded to five digits
    "original_dataset": "...",
    "prompt": messages[0]["content"].strip(),
    "transcript": messages[1]["content"].strip(),
    "soap_note": messages[2]["content"].strip(),
    "split": "sft" | "reward_model" | "grpo" | "validation" | "test",
}
```

The splitter validates that every retained record has exactly the expected
three message entries and that each has a usable `content` value. It applies
Python's `str.strip()` to each extracted value: leading and trailing spaces,
tabs, and line breaks are removed, while all internal whitespace and
formatting are preserved.

It drops the following source columns from derived files:

`split`, `task`, `input_modality`, `output_type`, `source_key`,
`source_file`, `has_audio`, `audio_format`, `audio_location`,
`audio_shard`, `audio_key`, `webdataset_key`, `messages`, and the original
`id` after using it to assign `unique_id` and the final split.

## Proposed Approach

1. Load the Hugging Face `train` and `validation` preview-table splits at an
   explicit revision, annotate their source origin, merge them, and filter
   `task == "asr2soap"`.
2. Validate unique non-empty source `id` values and the three-message
   structure.
3. Sort rows by source `id`, enumerate them to create stable zero-padded
   `unique_id` values, and sort the unique source IDs before applying
   `random.Random(42).shuffle`.
4. Allocate the shuffled IDs 60/15/15/5/5 using deterministic integer
   rounding, then add the assigned project value to the output `split`
   column.
5. Extract and trim `prompt`, `transcript`, and `soap_note` from the three
   messages; select only the derived output schema; write one local Parquet
   file per split plus `split_metadata.json`.

## Constraints

- Python 3.11 and a standard `venv`; Conda is not used.
- The splitter remains one script with a small number of functions.
- Source `id` and derived `unique_id` must each occur in exactly one final
  split.
- `unique_id` is a string to retain leading zeros; it is not an integer.
- Extracted text is normalized only with `str.strip()`; do not alter internal
  spaces, newlines, or other content.
- The normal public-data path needs no Hugging Face token.
- Generated files and credentials remain outside Git.

## Risks / Open Questions

| Item | Handling |
|---|---|
| Upstream dataset changes | Require explicit revision and record it in metadata |
| Message shape changes | Fail validation rather than silently extracting incorrect content |
| Current row count changes | Fail before assigning IDs if the filtered corpus is not 18,794 rows; change requires an explicit new split version |
| Group sizes vary | Allocation is by unique source `id` values; metadata reports final row counts |
| Large generated files | Keep them outside Git and recreate them on local/HPC/AWS as needed |

## Acceptance Criteria

- [ ] One script loads, merges, filters, transforms, splits, validates, and
      writes the derived data.
- [ ] Input is only source `train` and `validation`; output is only
      `asr2soap` records.
- [ ] Every source `id` is unique, non-empty, and assigned to one final
      split; `source_key` has no role in split assignment.
- [ ] Every output row exactly matches the derived output schema.
- [ ] Every extracted `prompt`, `transcript`, and `soap_note` has no leading
      or trailing whitespace, while internal formatting remains unchanged.
- [ ] `unique_id` values run consecutively from `"00001"` through
      `"18794"` for the current filtered corpus.
- [ ] The five `split` values follow the 60/15/15/5/5 allocation, subject
      only to deterministic integer rounding.
- [ ] Re-running with Python 3.11, the same source revision, seed, and
      script version produces the same assignments.
- [ ] Generated Parquet files and metadata are ignored by Git and can be
      recreated elsewhere.

## Validation

Use in-memory fixtures to test task filtering, source-ID uniqueness,
 three-message extraction and whitespace trimming, sequential ID assignment,
 split isolation, allocation, and stability under reordered input. Run Ruff
 and pytest, then run the script once with the pinned real revision and review
 the metadata and five output files.

## Rollback

Generated split files are disposable. Delete a failed output directory and
rerun with corrected code or inputs. A changed source revision, seed, output
schema, or expected row count defines a new split version and must not
overwrite the metadata for a validated run.
