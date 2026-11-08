"""Phase 3 (wrap-up): load the trained LoRA adapter in LLaMA Board's Chat tab and route a few tickets.

This is the UI-side sanity check after training: LLaMA Board loads Qwen3-1.7B-Base + the adapter
(Checkpoint path) with the same `default` chat template that was used for training, and we send
tickets in the training format ("Support Ticket: ...", same system prompt).

Prerequisite: LLaMA Board running locally (see scripts/llamaboard_train.py).

Usage:
    python scripts/llamaboard_chat.py --video-dir demo/raw
"""

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_common import SYSTEM_PROMPT, format_user  # noqa: E402

URL = "http://127.0.0.1:7860/"
CHECKPOINT = "train_2026-11-08-22-39-50"
VIEWPORT = {"width": 1440, "height": 900}
TICKETS = [
    ("Please create a new user account for the new employee starting Monday.", "Active Directory"),
    ("Outlook keeps asking me to re-enter my password and Teams won't sync.", "O365"),
    ("We're ready to retire [SERVER] — please remove it from monitoring.", "EOL"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def status(page: Page) -> str:
    return page.get_by_label("Textbox").filter(visible=True).first.input_value()


def wait_status(page: Page, needles: tuple[str, ...], timeout_s: int = 400) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = status(page)
        if any(n in s for n in needles):
            return s
        page.wait_for_timeout(2000)
    raise TimeoutError(f"status never contained {needles}; last={status(page)!r}")


def set_number(page: Page, label: str, value: str) -> None:
    box = page.get_by_label(f"number input for {label}").filter(visible=True).first
    box.fill(value)
    box.press("Enter")
    page.wait_for_timeout(300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens-dir", type=Path, default=Path("results/phase3_llamaboard"))
    ap.add_argument("--video-dir", type=Path, default=None)
    args = ap.parse_args()
    args.screens_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx_kwargs = dict(viewport=VIEWPORT)
        if args.video_dir:
            args.video_dir.mkdir(parents=True, exist_ok=True)
            ctx_kwargs.update(record_video_dir=str(args.video_dir), record_video_size=VIEWPORT)
        ctx = browser.new_context(**ctx_kwargs)
        page = ctx.new_page()
        page.goto(URL, wait_until="load")
        page.get_by_label("Model name", exact=True).wait_for(timeout=60_000)
        page.wait_for_timeout(2500)

        # Adapter = the Output Dir of the training run
        ckpt = page.get_by_label("Checkpoint path").filter(visible=True).first
        ckpt.click()
        page.wait_for_timeout(600)
        page.get_by_role("option", name=CHECKPOINT).first.click()
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)

        page.get_by_role("tab", name="Chat").click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Unload model").click()  # start from a clean state
        page.wait_for_timeout(3000)
        page.get_by_role("button", name="Load model", exact=True).click()
        log("loading base model + adapter …")
        t0 = time.time()
        wait_status(page, ("Model loaded",))
        log(f"model loaded in {time.time() - t0:.0f}s")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(args.screens_dir / "09_chat_adapter_loaded.png"))

        # Same system prompt as training; near-greedy decoding; short answers
        page.get_by_placeholder("System prompt (optional)").fill(SYSTEM_PROMPT)
        set_number(page, "Maximum new tokens", "16")
        set_number(page, "Temperature", "0.01")

        results = []
        for i, (ticket, expected) in enumerate(TICKETS, 1):
            page.get_by_role("button", name="Clear history").click()
            page.wait_for_timeout(800)
            page.get_by_placeholder("Input...").fill(format_user(ticket))
            page.wait_for_timeout(1200)
            page.get_by_role("button", name="Submit").click()
            bot = page.locator('[data-testid="bot"]')
            answer, t1 = "", time.time()
            while time.time() - t1 < 120:
                page.wait_for_timeout(1500)
                if bot.count():
                    now = bot.last.inner_text().strip()
                    if now and now == answer:  # stopped streaming
                        break
                    answer = now
            log(f"ticket {i}: expected={expected!r} answer={answer!r}")
            results.append({"ticket": ticket, "expected": expected, "answer": answer, "ok": answer == expected})
            # ticket (user bubble) and answer (bot bubble) in one view
            page.locator('[data-testid="user"]').last.evaluate("e => e.scrollIntoView({block: 'center'})")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(args.screens_dir / f"10_chat_ticket_{i}.png"))

        (args.screens_dir / "chat_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        page.get_by_role("button", name="Unload model").click()  # free memory for the next phase
        page.wait_for_timeout(4000)
        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
