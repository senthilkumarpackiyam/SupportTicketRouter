"""Phase 1 sanity check: confirm the training stack and an accelerator are usable.

Usage:  python scripts/check_env.py [--load-model]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true", help="also load the base model and generate a few tokens")
    parser.add_argument("--out", default="results/phase1_env_check.json")
    args = parser.parse_args()

    import accelerate
    import llamafactory
    import peft
    import sklearn
    import transformers

    device = pick_device()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "llamafactory": llamafactory.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device": device,
    }

    # Small matmul on the accelerator to prove kernels actually run.
    x = torch.randn(2048, 2048, device=device, dtype=torch.float16 if device != "cpu" else torch.float32)
    _ = x @ x  # warm-up
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = x @ x
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    report["matmul_tflops_fp16"] = round(10 * 2 * 2048**3 / elapsed / 1e12, 2)

    if args.load_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map=device).eval()
        report["model_load_seconds"] = round(time.perf_counter() - t0, 1)
        report["model_params_billion"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
        report["has_chat_template"] = tok.chat_template is not None

        prompt = "An IT helpdesk ticket about a broken printer should be routed to the"
        inputs = tok(prompt, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        report["generate_8_tokens_seconds"] = round(time.perf_counter() - t0, 2)
        report["sample_completion"] = prompt + tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
