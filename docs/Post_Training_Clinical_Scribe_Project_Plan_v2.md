# Post-Training an Open-Weight LLM for Automated Clinical Note Generation

## Project Goal

Build an end-to-end post-training pipeline for an open-weight LLM and
evaluate the impact of supervised fine-tuning (SFT) and reinforcement
learning (RL) on the quality of AI-generated clinical notes.

The project will compare three models:

1.  **Base GPT-OSS-20B**
2.  **LoRA Supervised Fine-Tuned GPT-OSS-20B**
3.  **LoRA SFT + GRPO GPT-OSS-20B**

The downstream task is:

> **Doctor--patient conversation → Clinical SOAP note**

This mirrors one of the most widely adopted real-world healthcare
applications of LLMs.

------------------------------------------------------------------------

# Motivation

Commercial AI medical scribes (Abridge, Microsoft Dragon Copilot, Nabla,
Suki, etc.) automate clinical documentation by converting
physician--patient conversations into structured clinical notes.

This project reproduces the modern post-training pipeline behind these
systems on publicly available datasets.

## Intended Use and Safety Boundary

This is a research project using publicly available data. Its models,
metrics, and any future frontend or demonstration are not intended for real
clinical use, medical decision-making, or deployment in patient-care
workflows. Generated notes must be treated as experimental artifacts, not
as clinical documentation.

No real patient data or unverified protected health information may be sent
to GPT-5 or any other external service. External annotation is limited to
approved project records after their synthetic or de-identification status
has been verified during dataset review.

------------------------------------------------------------------------

# Overall Pipeline

``` text
Clinical Dialogue
        │
        ▼
 Base GPT-OSS-20B
        │
        ▼
 LoRA Supervised Fine-Tuning
        │
        ▼
 Candidate Note Generation
        │
        ▼
 Preference Dataset Generation
        │
        ▼
 Reward Model Training
        │
        ▼
 GRPO Reinforcement Learning
        │
        ▼
 Final Clinical Scribe
```

------------------------------------------------------------------------

# Model Selection

**Primary LLM**

-   GPT-OSS-20B

Reasons:

-   Open-weight
-   Modern reasoning model
-   Fits within available HPC resources
-   Appropriate size for LoRA fine-tuning

------------------------------------------------------------------------

# Compute Resources

Development

-   MacBook Pro (M2 Pro)
-   Local inference
-   Dataset preprocessing
-   Frontend

Training

University HPC

-   \~48 GB GPU VRAM
-   Slurm scheduler

Supplementary AWS capacity

-   AWS credits are available and may supplement the HPC environment if
    they can be applied to a GPU-capable service (for example, GPU EC2 or
    a managed training service).
-   The project architecture and evaluation protocol remain unchanged.
-   Standard AWS Lambda functions are reserved, if useful, for bounded
    preprocessing, orchestration, or other short CPU tasks; they are not
    suitable for the long-running GPU SFT, reward-model, or GRPO training
    jobs. The exact AWS service, available GPU types, credit eligibility,
    and cost allocation remain to be confirmed before relying on this
    capacity.

Training techniques

-   QLoRA / LoRA
-   4-bit quantization where appropriate
-   Gradient accumulation

------------------------------------------------------------------------

# Stage 1 --- Supervised Fine-Tuning (SFT)

## Objective

Learn to generate high-quality clinical notes from physician--patient
conversations.

Training format

Input

``` text
Doctor-Patient Conversation
```

Output

``` text
Clinical SOAP Note
```

### Provisional Generation Prompt

The initial generation instruction is:

``` text
You are a medical scribe. Convert doctor-patient dialogues into structured
SOAP notes. S (Subjective): Patient history and symptoms. O (Objective):
Clinical findings and vitals. A (Assessment): Diagnoses and clinical
impressions. P (Plan): Treatments, labs, and follow-ups. Use professional
medical terminology, remain strictly factual, and mark missing info as
"N/A".
```

This prompt is provisional and will be refined after inspecting the
`asr2soap` examples and establishing the training chat template.

### Target Note Format

The desired output should match the structured, problem-oriented style of
the `asr2soap` example provided for this project:

1. `**1. Subjective**`, containing labeled `Chief Complaint (CC)`,
   `History of Present Illness (HPI)`, and, when supported, a structured
   `Review of Systems (ROS)`.
2. `**2. Objective**`, containing `Physical Examination` and
   `Tests/Results` subsections when those facts are present in the source
   dialogue.
3. `**3. Assessment and Plan**`, organized as numbered problems. Each
   problem is the top-level unit and contains an `Assessment` and a
   `Plan`. The plan may include labeled items such as medication,
   diagnostics, counseling, referral, and follow-up when applicable.

Use professional medical terminology and Markdown headings/bullets as in
the example. Every statement must be supported by the source dialogue;
missing information must be marked `N/A` rather than inferred. The
canonical top-level heading is the combined `**3. Assessment and Plan**`
section; separate top-level Assessment and Plan sections will not be used.

## Primary Dataset

### BeTraC2026-Augmented (`asr2soap`)

The primary training corpus will be the **`asr2soap`** task from the
BeTraC2026-Augmented dataset, with
[YapayNet/betrac2026-augmented](https://huggingface.co/datasets/YapayNet/betrac2026-augmented)
as the source of truth.

The BeTraC challenge's SynthDoPaCo corpus is documented as fully synthetic
in the TalTech BeTraC systems paper. The Hugging Face augmented collection
also identifies multiple original source datasets, so dataset exploration
must confirm the `original_dataset` composition of the selected `asr2soap`
records and the provenance/data-use terms of every included source before
sending any records to an external annotation service. The working
assumption is that the selected training content is synthetic; this
verification is a prerequisite for treating that assumption as confirmed.

Reasons:

-   Structured SOAP outputs closely matching modern AI clinical scribes.
-   Better aligned with ACI-Bench than dialogue-summary datasets.
-   Large enough to support SFT, reward-model generation, and RL.
-   Input/output format is already appropriate for transcript → SOAP
    generation.


### Data Split Strategy

The original BeTraC `train` and `validation` splits will be merged. The
combined data will be filtered to records where `task = asr2soap`, then
split deterministically by the source `id` field, which uniquely identifies
each dialogue/SOAP pair, into:

-   SFT
-   Reward Model
-   GRPO
-   Validation
-   Test

The initial split allocation is:

- SFT: 60%
- Reward Model: 15%
- GRPO: 15%
- Validation: 5%
- Test: 5%

No encounter may appear in more than one split. This allocation may be
revisited if later dataset size or training requirements warrant a change.

The split procedure must be reproducible: it will use the initial fixed,
version-controlled random seed `42`. The source `id` is the split unit;
`source_key` is not used because repeated values can still correspond to
distinct dialogues and SOAP summaries. Before shuffling, records are sorted
by source `id` and assigned deterministic zero-padded `unique_id` values
(`"00001"` through `"18794"` for the current filtered corpus). A post-split
validation must assert that every source `id` and `unique_id` occurs in
exactly one project split. The seed and dataset revision are experiment
inputs; any change to one creates a new data split version.

### Validated DATA-1 Split Version

The initial validated derived dataset version is `asr2soap_split_v2`. It is
defined by the following Git-tracked inputs, because generated `data/` files
are intentionally ignored and must be reproducible on local, HPC, or AWS
compute:

- Hugging Face dataset: `YapayNet/betrac2026-augmented`
- Immutable dataset revision: `cce6a4f2afc9203f491aec3a98d5c08121315dc5`
- Random seed: `42`
- Filtered corpus size: `18,794` records
- Split counts: SFT `11,277`; Reward Model `2,820`; GRPO `2,819`;
  Validation `939`; Test `939`

Recreate this derived cache with the committed DATA-1 splitter and Python
3.11:

```bash
python scripts/split_asr2soap.py \
  --revision cce6a4f2afc9203f491aec3a98d5c08121315dc5 \
  --output-dir data/asr2soap_split_v2
```

ACI-Bench is a small subset of BeTraC2026-Augmented (approximately 100
samples). It will not be held out or given separate external-evaluation
status. Its qualifying encounters are included in the same merged,
filtered corpus and may be assigned to any of the five encounter-disjoint
splits by the normal split procedure.


## Data Strategy

The derived split files will replace the source's nested `messages` field
with three explicit columns:

``` python
{
    "unique_id": "00001",
    "original_dataset": "...",
    "prompt": messages[0]["content"].strip(),
    "transcript": messages[1]["content"].strip(),
    "soap_note": messages[2]["content"].strip(),
    "split": "sft" | "reward_model" | "grpo" | "validation" | "test",
}
```

The splitter validates that each retained record has the expected three
message entries before extracting and trimming the outer whitespace from
these fields; it preserves their internal formatting. To keep derived files
small, it drops the source `split`, `task`, `input_modality`, `output_type`,
`source_key`, `source_file`, audio metadata, `webdataset_key`, nested
`messages`, and original `id` after deterministic `unique_id` assignment.

Train a LoRA adapter for one epoch as an initial proof of concept. 
This proof of concept is a code-and-compute validation, not a quality gate:
the data pipeline, training run, checkpointing, and validation-note
generation must complete successfully, and generated notes must satisfy the
target SOAP structure for at least 95% of validation examples. No
improvement over the Base model or GPT-5 comparison is required after this
epoch. After successful validation, continue SFT for at least two additional
epochs (three total initially) before evaluating model quality and
proceeding to later stages.

------------------------------------------------------------------------

# Stage 2 --- Preference Dataset Generation

A public preference dataset for clinical note generation does not
currently appear to exist.

Instead, generate one.

For each conversation:

Generate an initial set of four candidate notes from the **SFT checkpoint**
using different temperatures, random seeds, and decoding parameters. Four
is the initial annotation-cost/diversity tradeoff; increase the count only
if later analysis justifies it, with eight as the practical upper bound.

Result

``` text
Conversation

Candidate A

Candidate B
```

Use GPT-5 as an expert annotator.

The judge receives the conversation and anonymized candidate notes only;
candidate order must be randomized. The ground-truth note is not provided
to the judge. It remains available only for SFT and offline evaluation.
This avoids anchoring preferences to a single reference-writing style and
requires each candidate to be judged on evidence in the dialogue.

Prompt rubric

-   factual correctness against the conversation
-   unsupported claims and hallucinations
-   completeness, including missing symptoms or clinically relevant details
-   SOAP formatting and organization
-   clinical safety

Output

``` text
Preferred Note

Rejected Note

Explanation
```

The annotation protocol must allow a tie/abstain result when candidates are
meaningfully equivalent or cannot be reliably ranked. Tie/abstain examples
will be retained for analysis but excluded from the initial pairwise
reward-model training set.

For each encounter, GPT-5 produces one blinded ranking of all four
candidates rather than separate pairwise calls. The resulting strict
preferences are converted into every implied `chosen`/`rejected` pair (up
to six pairs per encounter) for reward-model training. Pair weights must
be normalized so that every encounter has equal total contribution to the
reward-model loss.

This produces pairwise preference data:

``` python
{
    "prompt": transcript,
    "chosen": preferred_note,
    "rejected": rejected_note
}
```

Training preferences are generated only from the Reward Model split. A
separate validation preference set is generated from the Validation split
using the same candidate-generation and GPT-5 ranking protocol. Validation
preferences are used only for reward-model and GRPO checkpoint selection;
they are never used for gradient updates.

------------------------------------------------------------------------

# Stage 3 --- Reward Model

Rather than using GPT-OSS as the reward model, train a much smaller
model.

Possible choices

-   Gemma 3 4B
-   Qwen 2.5 3B
-   Llama 3.x 3B

Input

``` text
Conversation

Candidate Note
```

Output

``` text
Scalar Reward
```

Training objective

Predict physician preference learned from generated pairwise
comparisons using the Bradley–Terry pairwise logistic loss. For a chosen
note and rejected note with scalar rewards `r_chosen` and `r_rejected`, the
model estimates `sigmoid(r_chosen - r_rejected)` as the probability that
the chosen note is preferred and minimizes
`-log(sigmoid(r_chosen - r_rejected))`. Pair weights are normalized per
encounter as specified in the preference-generation stage. Select the
reward-model checkpoint by pairwise accuracy on the GPT-5 validation
preferences; report Kendall's tau against the four-way rankings as a
secondary ranking diagnostic.

------------------------------------------------------------------------

# Stage 4 --- Reinforcement Learning

Algorithm

**GRPO**

Reasons

-   Simpler than PPO
-   Lower compute requirements
-   Supported by Hugging Face TRL
-   Appropriate for LoRA fine-tuning

The reward model supplies the scalar reward during optimization.

The learned reward model is the primary GRPO signal. Add only a small,
deterministic penalty for malformed notes that violate the required SOAP
structure (for example, missing required top-level headings). This
format-validity term is a guard against reward hacking; it does not judge
clinical correctness, completeness, or factuality, which remain the
responsibility of the learned reward model.

GRPO uses a third, disjoint encounter split. The policy only receives
the transcript as input; the reference SOAP note is never shown during
RL. During GRPO training, periodically generate notes for Validation
encounters from the current GRPO checkpoint and the frozen SFT checkpoint.
GPT-5 blindly compares those outputs from the transcript alone with
randomized output order. This direct validation preference is used for
GRPO checkpoint selection rather than relying solely on reward-model score.

The completed SFT checkpoint is also the frozen GRPO reference policy. Use
a KL-divergence penalty to constrain the RL policy from drifting too far
from the supervised transcript-to-SOAP behavior. Tune the KL coefficient on
the Validation split; its exact value is part of the deferred GRPO
hyperparameter configuration.

------------------------------------------------------------------------

# Evaluation

Evaluation is performed on the held-out BeTraC test split. ACI-Bench is
not a separate evaluation set because it is already incorporated into the
project's source corpus and split procedure.

All compared models use the same evaluation prompt, deterministic decoding
with temperature `0`, and the same maximum-output-token setting. Decoding
variation is limited to Stage 2 candidate generation. A small nonzero
temperature may be considered later for a real-world-style demonstration,
but it is outside the controlled evaluation protocol.

## Automatic Metrics

-   ROUGE
-   BERTScore
-   BLEURT (optional)

## Preference Metrics

Independent GPT-5 judge

On held-out test encounters, the judge receives the transcript and
anonymized outputs from the models being compared (Base, SFT, and SFT +
GRPO). Model-output order must be randomized. The reference note is not
provided, consistent with the preference-labeling protocol.

Evaluate

-   completeness
-   correctness
-   hallucination rate
-   formatting
-   organization

## Human Comparison

Structured clinical evaluation is deferred. A later qualitative clinical
review is anticipated for approximately 30 selected examples displayed in
the frontend. Reviewer recruitment, evaluation rubric, consent/oversight
requirements, and analysis protocol must be defined before that work
begins.

For selected examples compare

Base

↓

SFT

↓

SFT + RL

↓

Ground Truth

------------------------------------------------------------------------

# Frontend

Very lightweight frontend.

Dropdown

    Visit #12

Display

Left

-   Base GPT-OSS

Middle

-   LoRA SFT

Right

-   LoRA + GRPO

Bottom

-   Ground Truth SOAP Note

Show evaluation metrics beneath each output.

No backend required.

Static inference results are sufficient.

------------------------------------------------------------------------

# Expected Deliverables

## Training Code

-   Dataset preprocessing
-   LoRA SFT
-   Preference generation
-   Reward model training
-   GRPO training

## Evaluation

-   Automatic metrics
-   LLM-as-judge evaluation
-   Qualitative comparison

## Reproducibility Records

Every training and evaluation run must produce an immutable run record with
its configuration, Git revision, package environment, dataset-manifest
hash, model and checkpoint IDs, random seeds, metrics, and representative
generated outputs. These lightweight metadata, summaries, and samples are
intended to remain versioned in GitHub.

Model checkpoints, complete dataset copies, and full raw generation or
annotation outputs may be too large for the repository. Store those
externally and retain their versioned locations and integrity hashes in the
Git-tracked run record. The final external artifact store is deferred.

## Demo

Simple web interface comparing model outputs.

## Model Upload

Subject to confirmation of base-model and dataset license terms and any
applicable institutional requirements, upload the following models to
Hugging Face:
- SFT LoRA model
- SFT LoRA + GRPO RL Model
- Reward model

The full 20B parameter models do not need to be uploaded to HuggingFace, for example, we could just upload the LoRA adapter. Since GPT-OSS 20B model already exists on HF, using base model + those additional parameters will still allow for reproducibility. 

------------------------------------------------------------------------

# Future Extensions

-   Larger preference dataset
-   Human physician preference labels
-   Better reward model architectures
-   Multi-specialty fine-tuning
-   Longer-context encounters
-   Retrieval-augmented clinical note generation

------------------------------------------------------------------------

# Deferred Decisions

1.  Select the reward-model architecture and base checkpoint.
2.  Finalize SFT, reward-model, and GRPO hyperparameters.
3.  Pin the GPT-5 model/version, annotation prompt version, and annotation
    budget before preference generation begins.
4.  Confirm which AWS GPU-capable service, instance types, and credit
    eligibility can supplement the HPC environment.
5.  Define the later clinical-review protocol for approximately 30 frontend
    examples, including reviewers, rubric, and analysis method.
6.  Choose the external store for large artifacts and confirm public model
    release eligibility before any Hugging Face upload.

------------------------------------------------------------------------

# Dataset Split Philosophy

Each **unique encounter** belongs to exactly one split.

  -----------------------------------------------------------------------
  Split                           Purpose
  ------------------------------- ---------------------------------------
  SFT                             LoRA fine-tuning

  Reward Model                    Candidate generation + GPT preference
                                  labeling + reward model training

  GRPO                            Reinforcement learning with frozen
                                  reward model

  Validation                      Shared checkpoint selection and
                                  hyperparameter tuning across SFT, RM,
                                  and GRPO

  Test                            Final untouched benchmark
  -----------------------------------------------------------------------

The validation set is never used for gradient updates and is shared
across all training stages. It includes a separately generated preference
validation set for reward-model and GRPO checkpoint selection.

------------------------------------------------------------------------

# Final Pipeline

``` text
BeTraC2026-Augmented (asr2soap)
        │
 Merge train + validation
        │
 Deterministic split by source id (seed 42)
        │
 ┌──────┬─────────┬────────┬──────────┬───────┐
 │ SFT  │   RM    │  GRPO  │ Validation│ Test │
 └──────┴─────────┴────────┴──────────┴───────┘
     │        │         │
     ▼        ▼         ▼
  LoRA SFT  Reward Model  GRPO

Final evaluation compares:
- Base GPT-OSS
- LoRA SFT GPT-OSS
- LoRA SFT + GRPO GPT-OSS
```
