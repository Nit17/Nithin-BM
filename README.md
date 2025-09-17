# Nithin-BM

A personal Python learning repo with bite-sized, runnable examples.

## Folder structure

- `Python/` – Core Python concepts with runnable demos
  - `palindromes.py` – 4 palindrome approaches + demo
  - `lambda_examples.py` – Lambda functions in practice
  - `generators_iterators.py` – Generators, iterators, itertools
  - `decorators_examples.py` – Decorators: log/time/retry/cache
  - `context_managers.py` – Class-based and contextlib patterns
  - `asyncio_basics.py` – Async/await, gather, timeouts, semaphores
  - `test.py` – Scratch/test file
- `AIML/` – AI/ML learning
  - `finetune.py` – Fine-tuning overview (Transformers, LoRA/QLoRA, cloud)
  - `tensorflow_basics.py` – TF/Keras basics: tensors, small model, save/load

## Environments

Two virtual environments exist:
- `.venv/` – General Python work (3.13)
- `.venv312/` – TensorFlow-compatible environment (3.12)

Use the correct interpreter when running files (see commands below).

## How to run

Examples (adjust if your shell is different):

- Python examples:
  - `./.venv/bin/python Python/palindromes.py`
  - `./.venv/bin/python Python/lambda_examples.py`
  - `./.venv/bin/python Python/generators_iterators.py`
  - `./.venv/bin/python Python/decorators_examples.py`
  - `./.venv/bin/python Python/context_managers.py`
  - `./.venv/bin/python Python/asyncio_basics.py`

- AI/ML examples:
  - `./.venv/bin/python AIML/finetune.py` (prints guidance; no heavy deps required)
  - `./.venv312/bin/python AIML/tensorflow_basics.py` (requires TensorFlow)

## Notes

- Git identity for this repo is set to `Nit17 <nithinbm17@gmail.com>`.
- Large artifacts and generated outputs are ignored via `.gitignore` (e.g., `.keras`, `out-*`).
- If you want a different layout (e.g., `src/`), say the word and I’ll migrate files and update imports.
