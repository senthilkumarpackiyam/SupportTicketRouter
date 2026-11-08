"""Phase 3: Drive LLaMA Board (the LLaMA Factory web UI) in a real browser to launch LoRA training.

Exactly the course's manual steps, automated with Playwright so the run is reproducible and recorded:
  1. Model name  -> Qwen3-1.7B-Base   (model path + chat template auto-fill)
  2. Finetuning  -> lora (default)
  3. Dataset     -> support_tickets
  4. Preview dataset, Preview command (screenshots)
  5. Start, then wait for "Finished." while taking progress screenshots

Prerequisite: LLaMA Board running locally, from the LLaMA-Factory folder:
    GRADIO_SERVER_PORT=7860 llamafactory-cli webui

Usage:
    python scripts/llamaboard_train.py --dry-run          # fill the form + preview only
    python scripts/llamaboard_train.py --video-dir demo/raw  # fill, Start, wait, record video
"""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = "http://127.0.0.1:7860/"
MODEL_NAME = "Qwen3-1.7B-Base"
DATASET = "support_tickets"
VIEWPORT = {"width": 1440, "height": 900}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_dropdown(page: Page, label: str, value: str) -> None:
    """Gradio dropdown: focus the input, type to filter, click the exact option."""
    box = page.get_by_label(label, exact=True).filter(visible=True).first  # other tabs hold hidden twins
    box.scroll_into_view_if_needed()
    box.click()
    box.fill(value)
    page.wait_for_timeout(600)
    page.get_by_role("option", name=value, exact=True).first.click()
    page.wait_for_timeout(800)


def input_value(page: Page, label: str) -> str:
    # Textbox labels also contain their help text ("Model path\n\nPath to pretrained…"), so match by prefix.
    exact = page.get_by_label(label, exact=True).filter(visible=True)
    loc = exact if exact.count() else page.get_by_label(label).filter(visible=True)
    return loc.first.input_value()


def status_text(page: Page) -> str:
    """The status line LLaMA Board shows under the Output dir block ('Ready.', 'Running…', 'Finished.')."""
    return page.locator("body").inner_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fill the form and preview, but do not click Start")
    ap.add_argument("--screens-dir", type=Path, default=Path("results/phase3_llamaboard"))
    ap.add_argument("--video-dir", type=Path, default=None, help="record the browser session as .webm here")
    ap.add_argument("--timeout-min", type=float, default=120)
    args = ap.parse_args()
    args.screens_dir.mkdir(parents=True, exist_ok=True)
    shot = lambda page, name, full=False: page.screenshot(path=str(args.screens_dir / f"{name}.png"), full_page=full)  # noqa: E731

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
        shot(page, "01_llamaboard_home")
        log("LLaMA Board loaded")

        # 1-2. Model + finetuning method
        pick_dropdown(page, "Model name", MODEL_NAME)
        page.wait_for_timeout(1500)
        model_path = input_value(page, "Model path")
        template = input_value(page, "Chat template")
        method = input_value(page, "Finetuning method")
        log(f"model path={model_path!r}  template={template!r}  finetuning={method!r}")
        assert model_path == "Qwen/Qwen3-1.7B-Base", model_path
        assert method == "lora", method
        shot(page, "02_model_selected")

        # 3. Dataset
        pick_dropdown(page, "Dataset", DATASET)
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)
        page.get_by_label("Learning rate").filter(visible=True).first.scroll_into_view_if_needed()
        page.mouse.wheel(0, -250)
        page.wait_for_timeout(500)
        shot(page, "03_dataset_and_hparams")

        # 4a. Preview dataset (modal with the first ShareGPT records)
        page.get_by_role("button", name="Preview dataset").click()
        page.wait_for_timeout(2500)
        shot(page, "04_preview_dataset")
        page.get_by_role("button", name="Close").first.click()
        page.wait_for_timeout(800)

        # 4b. Preview command (the exact `llamafactory-cli train` call LLaMA Board will run)
        page.get_by_role("button", name="Preview command").click()
        page.wait_for_timeout(2500)
        output_dir = input_value(page, "Output dir")
        page.get_by_role("button", name="Preview command").scroll_into_view_if_needed()
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(600)
        shot(page, "05_preview_command")
        code = page.locator("code").filter(has_text="llamafactory-cli train")
        cmd_text = code.first.inner_text() if code.count() else ""
        (args.screens_dir / "preview_command.txt").write_text(cmd_text + "\n")
        log(f"output dir={output_dir!r}; preview command captured={bool(cmd_text)}")

        if args.dry_run:
            log("dry run: not starting training")
            ctx.close()
            browser.close()
            return

        # 5. Start training and wait for completion
        page.get_by_role("button", name="Start").click()
        t0 = time.time()
        log("Start clicked")
        page.wait_for_timeout(15_000)
        shot(page, "06_training_started")
        last_shot = 0.0
        milestones = {0.25: False, 0.5: False, 0.75: False}
        result = "timeout"
        while time.time() - t0 < args.timeout_min * 60:
            page.wait_for_timeout(20_000)
            text = status_text(page)
            if "Finished." in text:
                result = "finished"
                break
            if "Failed." in text or "Traceback" in text:
                result = "failed"
                break
            elapsed = time.time() - t0
            # keep the loss plot + progress bar in view
            page.get_by_role("button", name="Start").scroll_into_view_if_needed()
            if elapsed - last_shot > 120:
                shot(page, "07_training_progress_latest")
                last_shot = elapsed
                prog = page.locator("text=/\\d+\\/\\d+/").all_inner_texts()[:2]
                log(f"training… {elapsed / 60:.1f} min  progress={prog}")
                # milestone screenshots for the write-up / demo
                for m in milestones:
                    if not milestones[m] and prog:
                        try:
                            done, total = map(int, prog[0].split("/")[0:2])
                            if total and done / total >= m:
                                shot(page, f"07_training_{int(m * 100)}pct")
                                milestones[m] = True
                        except ValueError:
                            pass

        page.wait_for_timeout(3000)
        page.get_by_role("button", name="Start").scroll_into_view_if_needed()
        page.mouse.wheel(0, 250)
        page.wait_for_timeout(1000)
        shot(page, "08_training_finished" if result == "finished" else f"08_training_{result}")
        log(f"result={result} after {(time.time() - t0) / 60:.1f} min")
        (args.screens_dir / "run_summary.json").write_text(json.dumps({
            "result": result, "minutes": round((time.time() - t0) / 60, 1), "output_dir": output_dir,
            "model_path": model_path, "template": template, "finetuning": method, "dataset": DATASET,
        }, indent=2) + "\n")
        page.wait_for_timeout(4000)
        ctx.close()   # flushes the video file
        browser.close()


if __name__ == "__main__":
    main()
