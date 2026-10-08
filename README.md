# Text-to-Cypher LoRA Fine-Tuning

Fine-tune `Qwen2.5-3B-Instruct` to generate Neo4j Cypher queries from a graph
schema and a natural-language question. The training setup uses 4-bit loading
and Low-Rank Adaptation (LoRA) through Unsloth.

## Method

- Base model: `unsloth/Qwen2.5-3B-Instruct`
- Dataset: `tomasonjo/text2cypher-gpt4o-clean`
- Adaptation: LoRA, rank 16, alpha 16
- Quantization: 4-bit loading
- Hardware: CUDA-capable NVIDIA GPU
- Random seed: 42
- Evaluation design: database-level holdout

The databases `bluesky` and `stackoverflow2` are held out from training and
validation. This prevents examples from the same graph domain appearing in
both training and evaluation. The script writes the resulting splits to
`artifacts/` so the evaluation set can be preserved and inspected.

## Setup

Create a CUDA-enabled environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/train_lora.py
```

The default configuration trains on 800 examples, validates on 100 examples,
and reserves 100 examples from held-out databases for testing. Override these
values with `--train-size`, `--validation-size`, and `--test-size` when needed.

## Reproducibility

The split and training seed are fixed at `42`. Results should be reported with
the GPU model, CUDA/PyTorch versions, training configuration, and evaluation
script so that latency and accuracy comparisons are interpretable.

This repository currently contains the training pipeline. Exact-match scores,
LLM-judge results, and forgetting evaluations should be added only after they
are reproduced from the saved adapter and held-out test set.

## Project status

The original experiment was developed in Google Colab and then cleaned into a
standalone training script. Model checkpoints and private credentials are not
stored in this repository.
