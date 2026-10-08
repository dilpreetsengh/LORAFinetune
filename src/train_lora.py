"""Fine-tune Qwen2.5-3B-Instruct for schema-grounded Text-to-Cypher generation.

The split is database-level: the held-out databases are never used for training
or validation. This is intended to test generalization to unseen graph domains.
Run on a CUDA-enabled machine with the dependencies in requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


SEED = 42
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
HELD_OUT_DATABASES = {"bluesky", "stackoverflow2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs", help="Checkpoint directory")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=3)
    return parser.parse_args()


def format_example(row: dict) -> dict:
    instruction = (
        "You are a Cypher query expert. Given a Neo4j graph schema and a "
        "natural-language question, write the correct Cypher query.\n\n"
        f"Schema:\n{row['schema']}\n\n"
        f"Question: {row['question']}"
    )
    return {"instruction": instruction, "output": row["cypher"]}


def build_splits(dataset, train_size: int, validation_size: int, test_size: int):
    held_out = [row for row in dataset if row["database"] in HELD_OUT_DATABASES]
    remaining = [row for row in dataset if row["database"] not in HELD_OUT_DATABASES]

    rng = random.Random(SEED)
    rng.shuffle(held_out)
    rng.shuffle(remaining)

    if len(held_out) < test_size:
        raise ValueError("Not enough examples in the held-out databases for the test set.")
    if len(remaining) < train_size + validation_size:
        raise ValueError("Not enough examples in the training/validation pool.")

    test = held_out[:test_size]
    validation = remaining[:validation_size]
    train = remaining[validation_size : validation_size + train_size]
    return train, validation, test


def save_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def add_chat_format(example: dict, tokenizer) -> dict:
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 4-bit Unsloth training setup.")

    dataset = load_dataset("tomasonjo/text2cypher-gpt4o-clean", split="train")
    train, validation, test = build_splits(
        dataset, args.train_size, args.validation_size, args.test_size
    )

    train_rows = [format_example(row) for row in train]
    validation_rows = [format_example(row) for row in validation]
    test_rows = [format_example(row) for row in test]

    artifacts = Path("artifacts")
    save_json(train_rows, artifacts / "train.json")
    save_json(validation_rows, artifacts / "validation.json")
    save_json(test_rows, artifacts / "test.json")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    train_ds = Dataset.from_list(train_rows).map(
        lambda row: add_chat_format(row, tokenizer)
    )
    validation_ds = Dataset.from_list(validation_rows).map(
        lambda row: add_chat_format(row, tokenizer)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            report_to="none",
            seed=SEED,
        ),
    )

    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(validation_rows)}")
    print(f"Held-out test examples: {len(test_rows)}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
