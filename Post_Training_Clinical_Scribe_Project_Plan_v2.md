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

## Primary Dataset

### BeTraC2026-Augmented (`asr2soap`)

The primary training corpus will be the **`asr2soap`** task from the
BeTraC2026-Augmented dataset.

Reasons:

-   Structured SOAP outputs closely matching modern AI clinical scribes.
-   Better aligned with ACI-Bench than dialogue-summary datasets.
-   Large enough to support SFT, reward-model generation, and RL.
-   Input/output format is already appropriate for transcript → SOAP
    generation.

### ACI-Bench

ACI-Bench will be reserved primarily as the final evaluation benchmark.

### Data Split Strategy

The original BeTraC train and validation splits will be merged and
randomly split by **unique encounter** into:

-   SFT
-   Reward Model
-   GRPO
-   Validation
-   Test

No encounter may appear in more than one split.

### MTS-Dialog

Pros

-   High-quality
-   Already cleaned
-   Standard train/validation/test split

### MEDIQA-Chat

Pros

-   Research benchmark
-   Clinical note generation challenge

### ACI-Bench

Pros

-   Large
-   Realistic encounters
-   Multiple specialties

## Data Strategy

Normalize all datasets into

``` python
{
    "conversation": "...",
    "note": "..."
}
```

Train a LoRA adapter for one epoch as an initial proof of concept.

------------------------------------------------------------------------

# Stage 2 --- Preference Dataset Generation

A public preference dataset for clinical note generation does not
currently appear to exist.

Instead, generate one.

For each conversation:

Generate multiple candidate notes from the **SFT checkpoint** using
different temperatures, random seeds, and decoding parameters.

Result

``` text
Conversation

Ground Truth Note

Candidate A

Candidate B
```

Use GPT-5 as an expert annotator.

Prompt rubric

-   factual correctness
-   hallucinations
-   completeness
-   missing symptoms
-   SOAP formatting
-   organization

Output

``` text
Preferred Note

Rejected Note

Explanation
```

This produces pairwise preference data:

``` python
{
    "prompt": transcript,
    "chosen": preferred_note,
    "rejected": rejected_note
}
```

Preferences are generated only from the Reward Model split.

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
comparisons.

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

GRPO uses a third, disjoint encounter split. The policy only receives
the transcript as input; the reference SOAP note is never shown during
RL.

------------------------------------------------------------------------

# Evaluation

## Automatic Metrics

-   ROUGE
-   BERTScore
-   BLEURT (optional)

## Preference Metrics

Independent GPT-5 judge

Evaluate

-   completeness
-   correctness
-   hallucination rate
-   formatting
-   organization

## Human Comparison

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

## Demo

Simple web interface comparing model outputs.

------------------------------------------------------------------------

# Future Extensions

-   Larger preference dataset
-   Human physician preference labels
-   Better reward model architectures
-   Multi-specialty fine-tuning
-   Longer-context encounters
-   Retrieval-augmented clinical note generation

------------------------------------------------------------------------

# Open Questions

1.  Confirm the final SFT dataset(s).
2.  Determine whether preference generation should use one or multiple
    candidate pairs per encounter.
3.  Select the reward model architecture.
4.  Finalize GRPO hyperparameters.
5.  Determine evaluation benchmark and held-out test set.

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
across all training stages.

------------------------------------------------------------------------

# Final Pipeline

``` text
BeTraC2026-Augmented (asr2soap)
        │
 Merge train + validation
        │
 Shuffle by unique encounter
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
