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

## Concepts (Theory + Try it)

Quick links to each lesson with a one-line theory summary and how to run.

- Palindromes – Normalization choices; slicing vs two-pointer trade-offs.
  - File: `Python/palindromes.py`
  - Run: `./.venv/bin/python Python/palindromes.py`

- Lambdas – Tiny, anonymous single-expression functions; prefer `def` for anything non-trivial.
  - File: `Python/lambda_examples.py`
  - Run: `./.venv/bin/python Python/lambda_examples.py`

- Generators & Iterators – Lazy iteration; build memory-efficient pipelines with `yield` and `itertools`.
  - File: `Python/generators_iterators.py`
  - Run: `./.venv/bin/python Python/generators_iterators.py`

- Decorators – Wrap callables to add behavior; preserve metadata with `functools.wraps`.
  - File: `Python/decorators_examples.py`
  - Run: `./.venv/bin/python Python/decorators_examples.py`

- Context Managers – Ensure setup/teardown with `with`; use `__enter__/__exit__` or `@contextmanager`.
  - File: `Python/context_managers.py`
  - Run: `./.venv/bin/python Python/context_managers.py`

- Asyncio Basics – Single-threaded I/O concurrency with `async`/`await`; offload CPU-bound work.
  - File: `Python/asyncio_basics.py`
  - Run: `./.venv/bin/python Python/asyncio_basics.py`

- Hash maps (dicts) – Hashing, hashable keys, order guarantee, dynamic views, common pitfalls.
  - File: `Python/hashmaps_dicts.py`
  - Run: `./.venv/bin/python Python/hashmaps_dicts.py`

- Pattern Matching – Structural matching (match/case) for literals, sequences, mappings, classes, guards.
  - File: `Python/pattern_matching.py`
  - Run: `./.venv/bin/python Python/pattern_matching.py`

- Exceptions & Error Handling – Specific vs broad catches, EAFP vs LBYL, chaining, else/finally.
  - File: `Python/exceptions_handling.py`
  - Run: `./.venv/bin/python Python/exceptions_handling.py`

- Fine-tuning (NLP/LLMs) – When to fine-tune vs RAG/prompting; PEFT (LoRA/QLoRA) trade-offs and risks.
  - File: `AIML/finetune.py`
  - Run: `./.venv/bin/python AIML/finetune.py`

- TensorFlow Basics – Tensors, Keras models, `tf.data` pipelines; save/load with `.keras` format.
  - File: `AIML/tensorflow_basics.py`
  - Run: `./.venv312/bin/python AIML/tensorflow_basics.py`

## Environments

Two virtual environments exist:
- `.venv/` – General Python work (3.13)
- `.venv312/` – TensorFlow-compatible environment (3.12)
