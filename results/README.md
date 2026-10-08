# Recorded evaluation and limitations

Source: saved outputs in the uploaded FINETUNE.ipynb, preserved in the public sanitized notebook. These are original run observations, not independently rerun results.

## Text-to-Cypher

100 examples sampled from held-out bluesky and stackoverflow2 domains using seed 42. Exact match uses stripped strings, without semantic normalization or database execution. Fine-tuned: 10/100; base: 0/100. Both use greedy generation, up to 256 new tokens.

GPT-4o judging: fine-tuned preferred 68 times, base 28 times, 4 ties. A/B placement was randomized. The judge sees question, gold query, and two outputs, but not the full schema. The original parser treats unrecognized verdicts as ties; raw judge JSON is not uploaded, so the 4 reported ties cannot be independently classified. Inputs were paired using zip without explicit length/alignment assertions. For a future rerun, validate equal lengths and matching questions/gold queries, reject malformed verdicts separately, save A/B assignment and model metadata, and use schema/execution checks. The original code prints counts with percent signs; they equal percentages here only because there were 100 comparisons.

## General capability subset

200 questions from cais/mmlu, all configuration, test split, shuffled with seed 42. Fine-tuned 118/200 (59.0%); base 115/200 (57.5%). The 1.5 percentage-point difference is only 3 answers and does not establish a statistically significant improvement or absence of catastrophic forgetting. Original parsing checks whether generation starts with the gold letter; a future rerun should parse a single answer token and save per-question predictions.

## Training environment and run

Notebook logs record Tesla T4; Torch 2.11.0+cu128; CUDA Toolkit 12.8; Transformers 5.5.0; Unsloth 2026.8.3; Triton 3.6.0. The original installation logs also show TRL 0.24.0, datasets 4.3.0, bitsandbytes 0.50.0. These are recorded versions, not a tested lockfile. Successful training disables periodic checkpoint saves after an earlier serialization failure, then saves the adapter/tokenizer at the end. Runtime: 1:10:34, 300/300 steps.

| Step | Training loss | Validation loss |
|---|---:|---:|
| 50 | 0.588088 | 0.378088 |
| 100 | 0.035104 | 0.033213 |
| 150 | 0.019879 | 0.029878 |
| 200 | 0.024536 | 0.028916 |
| 250 | 0.021975 | 0.028960 |
| 300 | 0.021734 | 0.028692 |

Loss is not query accuracy. Training/validation domains overlap; only test domains are held out. Dataset revisions were not pinned. No documented rank/learning-rate sweep exists. Raw evaluation JSON and adapter weights remain outside this repo. Original generation warns about the missing attention mask; saved results are preserved, not retroactively changed.
