"""Phase 4: Review training: plot the loss curve from LLaMA Board's trainer_log.jsonl.

Same check as the course notebook ("Review Training — Loss Curve"): the loss should fall steadily
and then level off. On top of that it measures how flat the last epoch is, so "levelled off" is a
number rather than an impression.

Usage:   python scripts/phase4_loss_curve.py
Outputs: results/phase4_training_loss.png, results/phase4_training_review.json
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import viz_style as vs
from router_common import RESULTS_DIR, ROOT

ADAPTER_DIR = ROOT / "adapters" / "qwen3-1.7b-support-router-lora"


def load_log() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = [json.loads(l) for l in (ADAPTER_DIR / "trainer_log.jsonl").read_text().splitlines() if l.strip()]
    records = [r for r in records if r.get("loss") is not None]
    steps = np.array([r["current_steps"] for r in records])
    loss = np.array([r["loss"] for r in records])
    lr = np.array([r["lr"] for r in records])
    epoch = np.array([r["epoch"] for r in records])
    return steps, loss, lr, epoch


def course_verdict(final_loss: float) -> str:
    # identical thresholds to the course notebook
    if final_loss < 0.5:
        return "Loss is low — model has likely converged well."
    if final_loss < 1.2:
        return "Loss is moderate — model has learned but may benefit from more epochs."
    return "Loss is still high — consider more epochs, a lower learning rate, or checking the data format."


def plot(steps, loss, steps_per_epoch: int, path) -> None:
    vs.apply()
    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    for a in (ax, az):
        for e in (1, 2):
            a.axvline(e * steps_per_epoch, color=vs.AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)

    # Left: whole run
    ax.plot(steps, loss, color=vs.SERIES_1, marker="o", markersize=4, markeredgecolor=vs.SURFACE, markeredgewidth=1)
    ax.set_title("Training loss, whole run")
    ax.set_xlabel("Optimizer step (effective batch 16)")
    ax.set_ylabel("Cross-entropy loss (label tokens)")
    ax.set_xlim(0, steps.max() + 3)
    ax.set_ylim(0, loss.max() * 1.12)
    for e in range(3):
        ax.text(e * steps_per_epoch + steps_per_epoch / 2, loss.max() * 1.07, f"epoch {e + 1}",
                ha="center", va="center", fontsize=9, color=vs.MUTED)
    ax.annotate(f"{loss[0]:.2f}", (steps[0], loss[0]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=vs.INK, fontweight="bold")
    ax.annotate(f"{loss[-1]:.2f}", (steps[-1], loss[-1]), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9, color=vs.INK, fontweight="bold")

    # Right: zoom on epochs 2-3 (the part a full-range axis flattens)
    m = steps >= steps_per_epoch
    az.plot(steps[m], loss[m], color=vs.SERIES_1, marker="o", markersize=4, markeredgecolor=vs.SURFACE,
            markeredgewidth=1)
    last = steps > 2 * steps_per_epoch
    mean_last = loss[last].mean()
    az.axhline(mean_last, color=vs.INK_2, linewidth=0.9)
    az.text(steps[m].min() + 0.5, mean_last, f"epoch-3 mean {mean_last:.2f}", va="bottom", fontsize=9, color=vs.INK_2)
    az.set_title("Zoom: epochs 2–3 (levelling off)")
    az.set_xlabel("Optimizer step")
    az.set_ylim(0, max(0.4, loss[m].max() * 1.15))
    az.set_xlim(steps_per_epoch - 2, steps.max() + 2)

    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    steps, loss, lr, epoch = load_log()
    total_steps = int(steps.max())
    steps_per_epoch = int(round(total_steps / epoch.max()))
    train_results = json.loads((ADAPTER_DIR / "train_results.json").read_text())

    e = lambda k: loss[(steps > (k - 1) * steps_per_epoch) & (steps <= k * steps_per_epoch)]  # noqa: E731
    per_epoch = {f"epoch_{k}": {"first": round(float(e(k)[0]), 4), "last": round(float(e(k)[-1]), 4),
                                "mean": round(float(e(k).mean()), 4)} for k in (1, 2, 3)}
    # slope of a straight line through the epoch-3 points, in loss per 10 steps
    s3 = steps > 2 * steps_per_epoch
    slope = float(np.polyfit(steps[s3], loss[s3], 1)[0] * 10)

    review = {
        "logged_points": len(steps),
        "starting_loss_step_5": round(float(loss[0]), 4),
        "final_loss_step_90": round(float(loss[-1]), 4),
        "total_drop": round(float(loss[0] - loss[-1]), 4),
        "drop_pct": round(float(100 * (loss[0] - loss[-1]) / loss[0]), 1),
        "mean_train_loss_reported": round(train_results["train_loss"], 4),
        "per_epoch": per_epoch,
        "epoch3_slope_per_10_steps": round(slope, 4),
        "epoch3_min_max": [round(float(loss[s3].min()), 4), round(float(loss[s3].max()), 4)],
        "learning_rate_start_end": [float(lr[0]), float(lr[-1])],
        "course_verdict": course_verdict(float(loss[-1])),
        "validation_loss_available": False,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    plot(steps, loss, steps_per_epoch, RESULTS_DIR / "phase4_training_loss.png")
    (RESULTS_DIR / "phase4_training_review.json").write_text(json.dumps(review, indent=2) + "\n")

    print(f"Starting loss : {loss[0]:.4f}")
    print(f"Final loss    : {loss[-1]:.4f}")
    print(f"Total drop    : {loss[0] - loss[-1]:.4f}  ({review['drop_pct']}%)")
    print(json.dumps(per_epoch, indent=2))
    print(f"Epoch-3 slope : {slope:+.4f} per 10 steps   range {review['epoch3_min_max']}")
    print(review["course_verdict"])
    print(f"Saved -> {RESULTS_DIR}/phase4_training_loss.png, phase4_training_review.json")


if __name__ == "__main__":
    main()
