# Phase 1: Environment setup

Course equivalent: **"Install Dependencies"** + **"Check GPU environment"** cells.

## What was done

| Step | Result |
|---|---|
| Cloned LLaMA-Factory | commit `f6355ab` (2026-11-06), version `0.9.6.dev0` |
| Python environment | `.venv` created with `uv`, Python **3.11.13** (LLaMA-Factory requires ≥ 3.11) |
| Installed | `pip install -e ./LLaMA-Factory` + scikit-learn, seaborn, jupyter (see `requirements.txt`) |
| Accelerator check | Apple M2 GPU via PyTorch `mps` backend is available; fp16 matmul ≈ **3.15 TFLOPS** |
| Base model | `Qwen/Qwen3-1.7B-Base` downloaded to the HF cache (3.2 GB, **1.721 B** parameters) |
| Model smoke load | Loads to `mps` in fp16 in ~8 s; 8 greedy tokens in ~1.5 s |

Full machine-readable output: [`results/phase1_env_check.json`](../results/phase1_env_check.json).

### Why local Apple-GPU instead of Colab T4?

The course notebook uses a free Colab Tesla T4 (16 GB, CUDA). This machine is an Apple M2 with
16 GB of unified memory, no CUDA, but PyTorch's `mps` backend runs the same
Transformers/PEFT code on the Apple GPU. For this small task (468 short training tickets, LoRA,
1.7 B model) that is enough. It also lets every step be run, checked and recorded end-to-end from one
machine. The only CUDA-specific piece of the course setup, `bitsandbytes` (needed only for
4/8-bit QLoRA), is not installed and not needed for plain LoRA.

### What the base model does before fine-tuning

```
Prompt : An IT helpdesk ticket about a broken printer should be routed to the
Output : printer technician. True or False?\nA
```

The base model continues text but does not follow a routing instruction or stick to our label set.
That is the gap fine-tuning has to close.

## Issue found and fixed: hidden `.pth` files on macOS

After installing, `llamafactory-cli` failed with `ModuleNotFoundError: No module named 'llamafactory'`.

- **Cause:** on this Mac, files created inside the dot-folder `.venv` receive the macOS
  `UF_HIDDEN` file flag. Recent Python releases (3.11.13+) deliberately **skip hidden `.pth`
  files** at start-up. The editable install of LLaMA-Factory relies on exactly such a file
  (`_editable_impl_llamafactory.pth`), so the package became invisible.
- **Fix:** `chflags -R nohidden .venv`, now part of [`setup.sh`](../setup.sh).

## Chat template note (for Phases 3 and 5)

`Qwen3-1.7B-Base` ships the full Qwen3 chat template (`<|im_start|>system … <|im_end|>`, with
thinking-mode handling). LLaMA Board must train with the matching `qwen3` template, and
`classify()` must build prompts with the same template, or the router will see a different
prompt format at inference time than it was trained on. This will be checked explicitly.
