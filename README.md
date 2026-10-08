# Text-to-Cypher LoRA Fine-Tuning

Schema-grounded Neo4j query generation with Qwen2.5-3B-Instruct, Unsloth, 4-bit model loading, and LoRA.

## Recorded results

| Evaluation | Base | Fine-tuned |
|---|---:|---:|
| Strict exact match, 100 held-out queries | 0% | 10% |
| GPT-4o pairwise preference, 100 comparisons | 28 wins | 68 wins |
| MMLU subset, 200 questions | 57.5% | 59.0% |

The judge recorded 4 ties. Preference is not execution accuracy. See [evaluation notes](results/README.md) and [saved Colab outputs](notebooks/text_to_cypher_recorded.ipynb).

## Method

- NVIDIA Tesla T4, one GPU; 800 train / 100 validation / 100 test examples.
- Dataset: [tomasonjo/text2cypher-gpt4o-clean](https://huggingface.co/datasets/tomasonjo/text2cypher-gpt4o-clean), 15 database domains.
- Hold out bluesky and stackoverflow2 from both training and validation. Train and validation are example-level splits of the remaining database pool.
- Qwen2.5-3B-Instruct; rank 16, alpha 16, 4-bit loading, maximum sequence length 2048.
- Learning rate 2e-4; 3 epochs, 300 optimizer steps; effective batch size 8.
- 29,933,568 trainable parameters (0.96%); final validation loss 0.028692.

## Repository

- `src/train_lora.py`: cleaned standalone training entry point.
- `notebooks/text_to_cypher_recorded.ipynb`: original successful training, inference, exact-match, judge, and MMLU cells with saved outputs. Installation noise, duplicate failed attempts, and credentials removed. Outputs were not regenerated during cleanup.
- `results/README.md`: methodology, recorded environment, and limitations.

## Running

The notebook uses Colab-specific Drive paths. Your adapter and raw evaluation JSON files are required to rerun evaluation; they are not included in this repository. You do not need to retrain if the adapter is available.

```bash
pip install -r requirements.txt
python src/train_lora.py --output-dir outputs
```

Dependencies are unpinned; the recorded environment is documented in the results notes. Syntax has been checked, but the cleaned script has not been GPU-tested in a fresh environment. Unsloth/TRL APIs can vary by version. Do not assume latest package releases reproduce the recorded environment.

For judging, configure `OPENAI_API_KEY` securely in your environment. Never commit it. Running the judging cell sends questions and generated queries to OpenAI and incurs API costs.

## Scope and limitations

This is a student fine-tuning experiment, not a production database agent or custom CUDA-kernel project. Strict string matching does not capture equivalent query formulations. No execution-based query evaluation or hyperparameter sweep is documented. The 200-question MMLU result is a limited retention check, not proof of no catastrophic forgetting. Model and dataset terms still apply; no third-party license rights are implied.
