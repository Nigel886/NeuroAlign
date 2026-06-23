[OPEN]

# Debug Session: eval-llama-exit

## Symptom
- Running `python eval.py ...` exits abruptly during Llama-3 4-bit weight loading (~20–30%) without a Python traceback.

## Environment
- OS: Windows
- GPU: RTX 4070 8GB (WDDM)
- Conda env: align311 (Python 3.11)

## Hypotheses
- A: GPU OOM / driver reset causes hard process termination during 4-bit load.
- B: bitsandbytes / CUDA kernel incompatibility triggers native crash.
- C: Disk offload / HF cache IO error triggers termination without visible traceback.
- D: Python exception happens but is not surfaced (hook/output buffering).

## Evidence Plan
- Start Debug Server and add network-reported instrumentation around:
  - eval.py main() entry/exit + around `load_frozen_llm`
  - model.py load_frozen_llm() branch decisions + memory snapshots

## Status
- Confirmed by runtime evidence: the process dies inside `AutoModelForCausalLM.from_pretrained(...)`.
- Reproduced across three loader strategies:
  - 4-bit full GPU
  - 8-bit + CPU offload
  - fp16 + accelerate offload
- User selected option B: add explicit guardrails instead of continuing unsafe loading attempts.
- Next: raise a clear RuntimeError for `Llama-3 8B + 8GB-class GPU` or CPU-only paths.
