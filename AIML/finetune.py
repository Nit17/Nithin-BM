"""
Fine-tuning overview: local models and other options

This script is a safe, runnable guide that explains:
- What is fine-tuning and when to use it vs prompting/RAG
- Local fine-tuning with Hugging Face Transformers
  - Vanilla Trainer (e.g., text classification)
  - LoRA / QLoRA for parameter-efficient fine-tuning of LLMs
- Alternatives: cloud-managed fine-tuning and hosting options
- Practical tips: dataset prep, hyperparameters, eval, and exporting

By default, this script does NOT train models. It prints guidance and checks
for optional libraries. You can copy/paste the printed code snippets into your
own training environment when ready.
"""
from __future__ import annotations
import importlib.util as _importlib
import os
import sys
from textwrap import dedent
from typing import List


def _has_pkg(name: str) -> bool:
    return _importlib.find_spec(name) is not None


def print_intro() -> None:
    print(
        dedent(
            """
            What is fine-tuning?
            - You continue training a base model on your task-specific data so it adapts to your domain.

            When to fine-tune vs alternatives:
            - Prompting only: Great for quick results on general tasks; no training.
            - RAG (Retrieval-Augmented Generation): Keep data in a vector store and retrieve at query time; no model updates.
            - Fine-tuning: Best when the model must consistently perform a task/style without long prompts, or when latency/cost per request matters.
            """
        ).strip()
    )


def print_env_checklist() -> None:
    print("\nEnvironment checklist (local training):")
    print("- GPU recommended (NVIDIA + CUDA) for speed; CPU works but will be slow")
    print("- Python 3.10+; create a virtualenv")
    print("- Suggested packages:")
    print("  pip install transformers datasets accelerate evaluate")
    print("  pip install peft bitsandbytes  # for LoRA/QLoRA on LLMs")
    print("  pip install trl  # helpful for RLHF/PEFT workflows (optional)")


def demo_text_classification_scaffold() -> None:
    print("\nExample: Text classification with Transformers Trainer (vanilla)")
    print(
        dedent(
            """
            # Minimal scaffold (copy to your training env)
            from datasets import Dataset
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

            # 1) Data (replace with your dataset)
            samples = [
                {"text": "I love this!", "label": 1},
                {"text": "Terrible experience.", "label": 0},
            ]
            ds = Dataset.from_list(samples)

            # 2) Tokenizer & model
            model_name = "distilbert-base-uncased"
            tok = AutoTokenizer.from_pretrained(model_name)
            def tokenize(batch):
                return tok(batch["text"], truncation=True, padding=True)
            ds_tok = ds.map(tokenize, batched=True)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            # 3) Training
            args = TrainingArguments(
                output_dir="./out-clf",
                per_device_train_batch_size=16,
                num_train_epochs=1,
                learning_rate=2e-5,
                logging_steps=10,
                save_steps=50,
            )
            trainer = Trainer(model=model, args=args, train_dataset=ds_tok)
            trainer.train()

            # 4) Save
            trainer.save_model("./out-clf")
            tok.save_pretrained("./out-clf")
            """
        ).strip()
    )
    print("\nInstalled? transformers:", _has_pkg("transformers"), "datasets:", _has_pkg("datasets"))


def demo_lora_scaffold() -> None:
    print("\nExample: LoRA for causal LMs (parameter-efficient)")
    print(
        dedent(
            """
            # LoRA fine-tuning (PEFT) for a small causal LM (illustrative)
            # Requires: transformers, peft, datasets, accelerate
            from datasets import load_dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, TaskType

            model_name = "gpt2"  # or a small open LLM
            tok = AutoTokenizer.from_pretrained(model_name)
            tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name)
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["c_attn", "c_fc", "c_proj"],  # depends on model architecture
            )
            model = get_peft_model(model, peft_cfg)

            data = load_dataset("tiny_shakespeare")  # toy dataset
            def format_example(ex):
                return {"input_ids": tok(ex["text"], truncation=True, padding="max_length", max_length=256)["input_ids"]}
            ds = data["train"].map(format_example, remove_columns=data["train"].column_names)

            args = TrainingArguments(
                output_dir="./out-lora",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=2e-4,
                num_train_epochs=1,
                logging_steps=10,
                save_steps=100,
                fp16=True,
            )

            def data_collator(features):
                import torch
                ids = [f["input_ids"] for f in features]
                input_ids = torch.tensor(ids)
                labels = input_ids.clone()
                return {"input_ids": input_ids, "labels": labels}

            trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
            trainer.train()

            # Save PEFT adapters
            model.save_pretrained("./out-lora")
            tok.save_pretrained("./out-lora")
            """
        ).strip()
    )
    print("Installed? peft:", _has_pkg("peft"))


def demo_qlora_scaffold() -> None:
    print("\nExample: QLoRA (4-bit) to fit larger LLMs on a single GPU")
    print(
        dedent(
            """
            # QLoRA setup (illustrative). Requires: bitsandbytes, peft, accelerate, transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, TaskType

            model_name = "TheBloke/Llama-2-7B-GGUF"  # replace with an actual HF Transformers model that supports bnb int4

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )

            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_cfg, device_map="auto")

            lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05)
            model = get_peft_model(base, lora_cfg)

            # Prepare your dataset/tokenization similar to the LoRA example
            # Then train with Trainer just like before. Export adapters at the end.
            """
        ).strip()
    )
    print("Installed? bitsandbytes:", _has_pkg("bitsandbytes"))


def print_cloud_options() -> None:
    print("\nCloud / managed options to consider:")
    print("- Hugging Face AutoTrain Advanced: low-code fine-tuning + Spaces for hosting")
    print("- AWS SageMaker + JumpStart: scalable training/hosting with managed infra")
    print("- Google Vertex AI: custom training jobs, model registry, endpoints")
    print("- Azure ML: compute clusters, pipelines, and registries")
    print("- OpenAI/Anthropic/Google fine-tuning APIs: vendor-specific (closed models); simpler but model-locked")


def print_practical_tips() -> None:
    print("\nPractical tips:")
    print("- Start small: subset your data, 1-2 epochs, verify the loop")
    print("- Prefer parameter-efficient methods (LoRA/QLoRA) for LLMs to save VRAM and time")
    print("- Clean labels and balanced classes matter more than tiny hyper tweaks for classification")
    print("- Track metrics (accuracy/F1/perplexity) and use validation splits early")
    print("- Save tokenizer, model (or adapters), and training args for reproducibility")
    print("- For inference, you can merge LoRA adapters into the base model or load adapters at runtime")


def main(args: List[str]) -> int:
    print_intro()
    print_env_checklist()
    demo_text_classification_scaffold()
    demo_lora_scaffold()
    demo_qlora_scaffold()
    print_cloud_options()
    print_practical_tips()

    # Optional quick actions
    if len(args) > 1 and args[1] == "check":
        ok = all(_has_pkg(p) for p in ("transformers", "datasets", "peft", "bitsandbytes"))
        print("\nPackages available:")
        for p in ("transformers", "datasets", "peft", "bitsandbytes"):
            print(f"- {p}:", _has_pkg(p))
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
