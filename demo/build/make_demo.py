"""Phase 7: build the narrated demo video.

* Slides: HTML (1920x1080) rendered to PNG with Playwright; charts come straight from results/.
* Screen footage: the real LLaMA Board sessions recorded in Phase 3 (demo/recordings/).
* Voice: macOS text-to-speech, Indian English male voice "Aman" (fallback "Rishi"). Every
  sentence is synthesised separately, so the subtitle file has exact timings.
* Assembly: ffmpeg (bundled with the `imageio-ffmpeg` package) → H.264/AAC MP4.

Usage:   python demo/build/make_demo.py [--work /tmp/demo_work]
Outputs: demo/SupportTicketRouter_demo.mp4, demo/SupportTicketRouter_demo.srt, demo/narration_script.md
"""

import argparse
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
REC = ROOT / "demo" / "recordings"
OUT_MP4 = ROOT / "demo" / "SupportTicketRouter_demo.mp4"
OUT_SRT = ROOT / "demo" / "SupportTicketRouter_demo.srt"
OUT_MD = ROOT / "demo" / "narration_script.md"
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1920, 1080, 30, 48000
RATE = 172  # words per minute
LEAD, GAP, TAIL = 0.45, 0.32, 0.75  # seconds of silence: segment start, between sentences, segment end
REPO = "github.com/senthilkumarpackiyam/SupportTicketRouter"

# ── pronunciation fixes for the TTS voice (subtitles keep the written form) ───────────────
SPEAK = [
    (r"Qwen3", "Kwen 3"), (r"LoRA", "Lora"), (r"LLaMA", "Llama"), (r"O365", "Office 365"),
    (r"\bEOL\b", "E O L"), (r"5e-5", "5 times 10 to the minus 5"), (r"TF-IDF", "T F I D F"),
    (r"\bF1\b", "F 1"), (r"ShareGPT", "Share G P T"), (r"\bT4\b", "T 4"), (r"\bM2\b", "M 2"),
    (r"Fileservice", "File service"), (r"Colab", "Co-lab"), (r"%", " percent"), (r"80/20", "80, 20"),
]


def speakable(s: str) -> str:
    for pat, rep in SPEAK:
        s = re.sub(pat, rep, s)
    return s


# ── slide styling (same palette as the charts: scripts/viz_style.py) ─────────────────────
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--blue:#2a78d6;
--orange:#eb6834;--soft:#eef4fc;--soft2:#fdf0ea;--green:#1f8a4c;--red:#c23b22}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1920px;height:1080px;background:var(--bg);color:var(--ink);overflow:hidden;
font-family:-apple-system,"SF Pro Display","Helvetica Neue",Helvetica,Arial,sans-serif}
.slide{position:relative;width:1920px;height:1080px;padding:84px 120px 120px}
.kicker{font-size:25px;font-weight:650;color:var(--blue);letter-spacing:.06em;text-transform:uppercase}
h1{font-size:64px;line-height:1.08;font-weight:760;margin-top:16px;letter-spacing:-.015em}
h2{font-size:34px;font-weight:700;margin-bottom:18px}
.lede{font-size:31px;color:var(--ink2);margin-top:20px;line-height:1.42;max-width:1560px}
.footer{position:absolute;left:120px;right:120px;bottom:40px;display:flex;justify-content:space-between;
font-size:20px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}
.row{display:flex;gap:56px;margin-top:44px;align-items:flex-start}
.col{flex:1}
.chips{display:flex;gap:22px;margin-top:40px;flex-wrap:wrap}
.chip{background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px 28px;min-width:260px}
.chip b{display:block;font-size:46px;font-weight:760;letter-spacing:-.01em}
.chip span{font-size:22px;color:var(--ink2)}
ul.clean{list-style:none;font-size:29px;line-height:1.5;color:var(--ink)}
ul.clean li{padding:9px 0 9px 34px;position:relative;border-bottom:1px solid var(--line)}
ul.clean li:before{content:"";position:absolute;left:4px;top:24px;width:12px;height:12px;border-radius:3px;background:var(--blue)}
table{border-collapse:collapse;width:100%;font-size:26px}
td,th{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:20px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.ok{color:var(--green);font-weight:700}.bad{color:var(--red);font-weight:700}
.img{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;display:flex;justify-content:center}
.img img{max-width:100%;max-height:100%;object-fit:contain}
.callout{background:var(--soft);border-left:6px solid var(--blue);border-radius:10px;padding:22px 26px;font-size:26px;
line-height:1.45;color:var(--ink)}
.callout.warn{background:var(--soft2);border-left-color:var(--orange)}
.mono{font-family:"SF Mono",Menlo,monospace;font-size:.9em}
.flow{display:flex;align-items:center;gap:18px;margin-top:10px}
.box{background:#fff;border:2px solid var(--line);border-radius:16px;padding:22px 24px;font-size:26px;text-align:center;line-height:1.35}
.box.hl{border-color:var(--blue);background:var(--soft)}
.arrow{font-size:40px;color:var(--muted)}
.slide.center{display:flex;flex-direction:column;justify-content:center;padding-bottom:150px}
.slide.center h1.hero{font-size:104px}
.slide.center .lede{font-size:38px}
.big table{font-size:31px} .big .box{font-size:31px;padding:28px 30px} .big .callout{font-size:30px}
.bar{height:34px;border-radius:0 6px 6px 0;background:var(--line)}
.bar.base{background:#f3c2ab}.bar.ft{background:var(--blue)}.bar.cl{background:#b9b8b0}
"""


def slide(kicker: str, title: str, body: str, center: bool = False, hero: bool = False,
          foot_left: str = "Support Ticket Router · The Gen Academy, Week 5") -> str:
    cls = "slide center" if center else "slide"
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="{cls}"><div class="kicker">{kicker}</div><h1 class="{'hero' if hero else ''}">{title}</h1>{body}
<div class="footer"><span>{foot_left}</span><span>{REPO}</span></div></div></body></html>"""


def img(name: str, h: int) -> str:
    return f'<div class="img" style="height:{h}px"><img src="{(RES / name).as_uri()}"></div>'


def frame_html(caption: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
.hdr{{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 40px;border-bottom:1px solid var(--line)}}
.hdr b{{font-size:26px;font-weight:700}} .hdr span{{font-size:22px;color:var(--muted)}}
.tag{{color:var(--blue);font-weight:700;margin-right:14px;font-size:22px;letter-spacing:.06em;text-transform:uppercase}}
</style></head><body><div class="hdr"><div><span class="tag">Live screen recording</span><b>{caption}</b></div>
<span>Support Ticket Router</span></div></body></html>"""


# ── the storyboard ────────────────────────────────────────────────────────────────────────
def storyboard() -> list[dict]:
    labels = [("Active Directory", "Identity &amp; access"), ("Computer-Services", "Devices, printers"),
              ("EOL", "Server retirement"), ("Fileservice", "Shares &amp; permissions"),
              ("O365", "Outlook, Teams, mailbox"), ("Software", "Installs &amp; licences"),
              ("Support general", "Human triage")]
    label_rows = "".join(f"<tr><td><b>{a}</b></td><td style='color:#52514e'>{b}</td></tr>" for a, b in labels)
    smoke = [("Please create a new user account for the new employee starting Monday.", "Active Directory", "Support general", False),
             ("I cannot access the shared network folder — Access Denied.", "Fileservice", "Fileservice", True),
             ("Outlook keeps asking me to re-enter my password, Teams won't sync.", "O365", "O365", True),
             ("We're ready to retire [SERVER] — remove it from monitoring.", "EOL", "EOL", True),
             ("Adobe Acrobat fails to install, error code 1603.", "Software", "Software", True)]
    smoke_rows = "".join(
        f"<tr><td style='font-size:22px'>{t}</td><td>{e}</td><td class='{'ok' if ok else 'bad'}'>{p} {'✓' if ok else '✗'}</td></tr>"
        for t, e, p, ok in smoke)
    ctx = [("Majority class", 25.6, "cl"), ("Base Qwen3-1.7B (A–G)", 24.8, "base"), ("TF-IDF nearest neighbour", 64.1, "cl"),
           ("TF-IDF + logistic regression", 69.2, "cl"), ("Fine-tuned Qwen3-1.7B (LoRA)", 72.6, "ft")]
    ctx_rows = "".join(
        f"<tr><td style='white-space:nowrap;padding-right:24px'>{n}</td><td style='width:100%'><div style='display:flex;align-items:center;gap:14px'>"
        f"<div class='bar {c}' style='width:{v * 5.4:.0f}px'></div><b>{v:.1f}%</b></div></td></tr>" for n, v, c in ctx)

    return [
        dict(kind="slide", name="01_title", html=slide(
            "The Gen Academy · Week 5 project", "Support Ticket Router",
            """<p class="lede">Fine-tuning <b>Qwen3-1.7B-Base</b> with a <b>LoRA</b> adapter in <b>LLaMA Factory</b>
            to send every IT support ticket to the right queue.</p>
            <div class="chips"><div class="chip"><b>585</b><span>labelled tickets</span></div>
            <div class="chip"><b>7</b><span>resolver queues</span></div>
            <div class="chip"><b>1 label</b><span>out, never a reply</span></div>
            <div class="chip"><b>72.6%</b><span>accuracy on held-out tickets</span></div></div>""", center=True, hero=True),
             say=["Hello, and welcome to this demo of the Support Ticket Router.",
                  "In this project, a small open model, Qwen3 1.7 billion, is fine-tuned with LoRA, using LLaMA Factory, "
                  "to route internal IT support tickets to the right team.",
                  "This is the Week 5 project of The Gen Academy."]),
        dict(kind="slide", name="02_business", html=slide(
            "The business case", "Route first, read later",
            f"""<div class="row big"><div class="col" style="flex:1.1">
            <div class="flow"><div class="box">Ticket<br><span style="font-size:21px;color:#52514e">email · Slack · portal</span></div>
            <div class="arrow">→</div><div class="box hl"><b>Router</b><br><span style="font-size:21px">fine-tuned 1.7B model</span></div>
            <div class="arrow">→</div><div class="box">1 of 7<br>queues</div></div>
            <div class="callout" style="margin-top:44px">The model <b>never writes a reply</b>. It predicts one label and a webhook
            does the rest. That's much cheaper and faster than a frontier model for every ticket, and more robust than keyword rules.</div>
            </div><div class="col" style="flex:.9"><table><tr><th>Label</th><th>Queue</th></tr>{label_rows}</table></div></div>""",
            center=True),
             say=["An IT help desk receives tickets by email, by Slack, and through a help portal.",
                  "Today, a person reads every single ticket just to decide which queue it belongs to.",
                  "The router reads the ticket first, and answers with exactly one of seven labels.",
                  "It never writes a reply. It only routes, and the rest of the tooling takes over.",
                  "A small fine-tuned model is much cheaper and faster for this than calling a frontier model for every ticket."]),
        dict(kind="slide", name="03_data", html=slide(
            "Phase 2 · Prepare the dataset", "585 tickets, stratified 80/20 split",
            f"""<div class="row"><div class="col" style="flex:.8"><ul class="clean">
            <li><b>468</b> training tickets, <b>117</b> held out for validation</li>
            <li>Converted to <b>ShareGPT</b> chat records: system prompt, ticket, label</li>
            <li>Registered in LLaMA Factory as <span class="mono">support_tickets</span></li>
            <li>No train/val overlap; longest example 183 tokens</li>
            <li>Verified which tokens are trained: only the label</li></ul></div>
            <div class="col" style="flex:1.2">{img("phase2_class_distribution.png", 560)}</div></div>"""),
             say=["The dataset has 585 labelled tickets.",
                  "We made a stratified 80/20 split: 468 tickets for training, and 117 held out for validation, which the model never sees.",
                  "The training tickets are converted to ShareGPT chat format, and registered in LLaMA Factory as the support tickets dataset.",
                  "Before training, we checked for leakage between the splits, and verified exactly which tokens the model learns from."]),
        dict(kind="clip", name="04_llamaboard_setup", src="llamaboard_train_session.mp4", start=0.0, end=21.5,
             caption="Phase 3 · LLaMA Board Train tab, driven by a Playwright script", max_slow=1.6,
             say=["Training happens in LLaMA Board, the web interface of LLaMA Factory.",
                  "Here, a Playwright script drives the browser, so every click is reproducible.",
                  "It selects the Qwen3 1.7 billion base model, LoRA as the fine-tuning method, and the support tickets dataset.",
                  "It previews the data and the exact training command, and then presses Start.",
                  "We kept LLaMA Board's defaults: learning rate 5e-5, three epochs, LoRA rank 8, and an effective batch size of 16."]),
        dict(kind="clip", name="05_training_timelapse", src="llamaboard_train_session.mp4", start=200.0, end=1498.0,
             caption="Phase 3 · Training in progress, first epoch (time-lapse)", fit=True,
             say=["Instead of a Colab T4, training ran locally on an Apple M2 laptop GPU.",
                  "Here is the first epoch as a time-lapse. The loss curve in LLaMA Board drops quickly.",
                  "Only 8.7 million parameters are trained. That is half a percent of the model."]),
        dict(kind="slide", name="06_loss", html=slide(
            "Phase 4 · Review training", "Loss falls 97%, then levels off",
            f"""<div style="margin-top:34px">{img("phase4_training_loss.png", 520)}</div>
            <div class="chips" style="margin-top:28px"><div class="chip"><b>7.16 → 0.19</b><span>training loss</span></div>
            <div class="chip"><b>90 steps</b><span>3 epochs · 69 min on Apple M2</span></div>
            <div class="chip"><b>8.7 M</b><span>trainable parameters (0.5%)</span></div>
            <div class="chip"><b>Flat</b><span>in epoch 3 → no retrain</span></div></div>"""),
             say=["The full run took 90 steps and about 69 minutes.",
                  "The loss falls from 7.16 to 0.19, a drop of 97%, and it stays flat during the third epoch.",
                  "So the model has converged, and there was no need to retrain."]),
        dict(kind="clip", name="07_llamaboard_chat", src="llamaboard_chat_session.mp4", start=0.0, end=40.5,
             caption="Phase 3 · LLaMA Board Chat tab: base model + trained LoRA adapter", max_slow=1.3, max_fast=1.5,
             say=["Back in LLaMA Board, the Chat tab loads the base model together with the trained adapter.",
                  "We send tickets in the same format used for training, and the model answers with a single label.",
                  "The first ticket, a new user account request, comes back as Support general. We will come back to that.",
                  "The Outlook and Teams problem goes to O365, and retiring a server goes to EOL. Both are correct."]),
        dict(kind="slide", name="08_merge_smoke", html=slide(
            "Phase 5 · Merge + smoke test", "One standalone model, 4 of 5 smoke tickets",
            f"""<div class="row" style="margin-top:36px"><div class="col" style="flex:.62"><ul class="clean">
            <li>Adapter folded into the base weights (<span class="mono">merge_and_unload</span>)</li>
            <li>392 / 392 LoRA tensors loaded</li><li>Merged output = adapter output</li>
            <li>3.45 GB standalone model</li></ul>
            <div class="callout warn" style="margin-top:30px">Training tickets asking for a <b>new user account</b> are labelled
            <b>Fileservice 9×</b>, Support general 3×, Active Directory only 2×. The model learned the data's convention.</div></div>
            <div class="col" style="flex:1.38"><table><tr><th>Smoke ticket</th><th>Expected</th><th>Fine-tuned</th></tr>{smoke_rows}</table></div></div>"""),
             say=["Next, the adapter is merged into the base weights, to give one standalone model.",
                  "We checked that the merged model gives the same output as the adapter.",
                  "On the course's five smoke test tickets, it gets four right.",
                  "The miss is interesting. In the training data, new user account requests are labelled Fileservice nine times, "
                  "Support general three times, and Active Directory only twice.",
                  "So the model simply learned the convention in the data."]),
        dict(kind="slide", name="09_results", html=slide(
            "Phase 6 · Evaluate on 117 held-out tickets", "24.8% → 72.6% accuracy",
            f"""<div style="margin-top:30px">{img("phase6_baseline_vs_finetuned.png", 600)}</div>
            <div class="chips" style="margin-top:24px"><div class="chip"><b>+47.9 pts</b><span>accuracy gain</span></div>
            <div class="chip"><b>0.08 → 0.68</b><span>macro F1</span></div>
            <div class="chip"><b>117 / 117</b><span>clean label outputs</span></div></div>"""),
             say=["Now the real test: all 117 held-out tickets.",
                  "The base model, given the seven labels as multiple choice from A to G, reaches only 24.8%. It mostly just picks option A.",
                  "The fine-tuned model reaches 72.6% accuracy. That is a gain of 48 points, and the macro F1 score rises from 0.08 to 0.68."]),
        dict(kind="slide", name="10_confusion", html=slide(
            "Phase 6 · Where it works, where it doesn't", "Strong on EOL and Fileservice, weak on Active Directory",
            f"""<div class="row" style="margin-top:30px"><div class="col" style="flex:1.05">{img("phase6_confusion_matrix.png", 700)}</div>
            <div class="col" style="flex:.95"><ul class="clean">
            <li><b>EOL</b> 12 / 12, perfect</li><li><b>Fileservice</b> 23 / 28, 92% precision</li>
            <li><b>Active Directory</b> only 2 / 9 caught</li><li>18 of 32 errors drift to <b>Support general</b></li></ul>
            <div class="callout warn" style="margin-top:30px"><b>Course notebook bug fixed:</b> <span class="mono">classification_report</span>
            was called without <span class="mono">labels=</span>, so every row carried the wrong class name.</div></div></div>"""),
             say=["The confusion matrix shows where the model works, and where it does not.",
                  "EOL is perfect, 12 out of 12, and Fileservice is strong.",
                  "The weak spot is Active Directory: only 2 of 9 tickets are caught. Most errors drift to the catch-all Support general queue.",
                  "Along the way, we also found and fixed a bug in the course notebook. Its classification report was called without "
                  "the labels argument, so every row had the wrong class name."]),
        dict(kind="slide", name="11_confidence", html=slide(
            "Phase 6 · Putting it to work", "Auto-route only confident tickets",
            f"""<div class="row" style="margin-top:30px"><div class="col" style="flex:1.25">{img("phase6_confidence_thresholds.png", 600)}</div>
            <div class="col" style="flex:.75"><div class="chips" style="flex-direction:column;margin-top:0">
            <div class="chip"><b>44%</b><span>of tickets auto-routed at confidence ≥ 0.8</span></div>
            <div class="chip"><b>86%</b><span>accuracy on those tickets</span></div>
            <div class="chip"><b>278 ms</b><span>median per ticket, Apple M2 GPU</span></div></div></div></div>"""),
             say=["The model's confidence score is genuinely useful.",
                  "If we route automatically only when the confidence is at least 0.8, 44% of tickets are handled at 86% accuracy, "
                  "and the rest go to a person.",
                  "Each prediction takes about 280 milliseconds on the laptop GPU."]),
        dict(kind="slide", name="12_context", html=slide(
            "Honest context · next steps", "Good router, small test set",
            f"""<div class="row" style="margin-top:36px"><div class="col" style="flex:1.15"><h2>Accuracy, same 117 tickets</h2>
            <table>{ctx_rows}</table><p style="font-size:22px;color:#52514e;margin-top:14px">95% interval ≈ ±8 points with 117 tickets</p></div>
            <div class="col" style="flex:.85"><h2>Next steps</h2><ul class="clean">
            <li>Write a labelling rule for account requests and relabel</li>
            <li>Auto-route only confident predictions</li>
            <li>Second run with a higher learning rate (1e-4 to 2e-4) and a validation split</li></ul></div></div>"""),
             say=["For honest context, a simple TF-IDF and logistic regression model reaches 69% on the same split.",
                  "With only 117 test tickets, the fine-tuned model's lead is within the margin of error.",
                  "The next steps are clear: agree on a labelling rule for account requests and relabel them, "
                  "auto-route only confident predictions, and try a second training run with a higher learning rate."]),
        dict(kind="slide", name="13_close", html=slide(
            "Thank you", "Everything is on GitHub",
            f"""<p class="lede" style="font-size:40px;color:#2a78d6;font-weight:650">{REPO}</p>
            <div class="row"><div class="col"><ul class="clean">
            <li>Reproducible scripts for all 7 phases</li><li>The trained LoRA adapter (35 MB)</li>
            <li>Executed, Colab-ready notebook</li></ul></div><div class="col"><ul class="clean">
            <li>LLaMA Board screenshots and recordings</li><li>Metrics, charts, error analysis</li>
            <li>A write-up for every phase</li></ul></div></div>""", center=True),
             say=["Everything is in the GitHub repository: the code, the trained adapter, the executed notebook, "
                  "and a write-up for every phase.",
                  "Thank you for watching!"]),
    ]


# ── helpers ───────────────────────────────────────────────────────────────────────────────
def run(args: list) -> None:
    subprocess.run([str(a) for a in args], check=True, capture_output=True)


def pick_voice() -> str:
    voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    for v in ("Aman (English (India))", "Rishi"):
        if any(line.startswith(v) for line in voices.splitlines()):
            return v
    raise RuntimeError("No Indian English male voice (Aman / Rishi) installed")


def tts(text: str, voice: str, out_wav: Path) -> float:
    aiff = out_wav.with_suffix(".aiff")
    run(["say", "-v", voice, "-r", RATE, "-o", aiff, speakable(text)])
    run([FF, "-y", "-i", aiff, "-ar", SR, "-ac", 1, "-sample_fmt", "s16", out_wav])
    with wave.open(str(out_wav)) as w:
        return w.getnframes() / w.getframerate()


def build_audio(sentences: list[str], voice: str, work: Path, name: str) -> tuple[Path, float, list[tuple[float, float, str]]]:
    pcm, cues, t = bytearray(), [], 0.0
    silence = lambda s: b"\x00\x00" * int(SR * s)  # noqa: E731
    pcm += silence(LEAD)
    t += LEAD
    for i, s in enumerate(sentences):
        wav = work / f"{name}_{i:02d}.wav"
        d = tts(s, voice, wav)
        with wave.open(str(wav)) as w:
            pcm += w.readframes(w.getnframes())
        cues.append((t, t + d, s))
        t += d
        gap = GAP if i < len(sentences) - 1 else TAIL
        pcm += silence(gap)
        t += gap
    out = work / f"{name}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(pcm))
    return out, t, cues


def render_pngs(pages: dict[str, str], work: Path) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        for name, html in pages.items():
            f = work / f"{name}.html"
            f.write_text(html)
            pg.goto(f.as_uri(), wait_until="load")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(work / f"{name}.png"))
        b.close()


ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-r", FPS,
       "-c:a", "aac", "-b:a", "160k", "-ar", SR, "-ac", 2]


def fades(d: float) -> str:
    return f"fade=t=in:st=0:d=0.35,fade=t=out:st={d - 0.35:.3f}:d=0.35"


def build_slide(png: Path, wav: Path, d: float, out: Path) -> None:
    run([FF, "-y", "-loop", 1, "-framerate", FPS, "-i", png, "-i", wav, "-t", f"{d:.3f}",
         "-vf", f"format=yuv420p,{fades(d)}", "-tune", "stillimage", *ENC, out])


def build_clip(seg: dict, frame_png: Path, wav: Path, d: float, out: Path) -> tuple[float, float]:
    """Returns (playback speed, segment duration). If the footage outlasts the narration even at
    the maximum speed-up, the segment is extended (silence) so no footage is cut."""
    clip_len = seg["end"] - seg["start"]
    if seg.get("fit"):
        k = d / clip_len  # time-lapse: squeeze the whole range into the narration
    else:
        k = min(max(d / clip_len, 1 / seg.get("max_fast", 1.0)), seg.get("max_slow", 1.0))
    shown = clip_len * k
    d = max(d, shown + 0.6)
    pad = max(0.0, d - shown)
    vf = (f"[1:v]setpts=(PTS-STARTPTS)*{k:.5f},fps={FPS},scale=1600:1000:flags=lanczos,"
          f"tpad=stop_mode=clone:stop_duration={pad + 1:.3f}[v];"
          f"[0:v][v]overlay=160:72:shortest=0,format=yuv420p,{fades(d)}[out]")
    run([FF, "-y", "-loop", 1, "-framerate", FPS, "-i", frame_png,
         "-ss", seg["start"], "-to", seg["end"], "-i", REC / seg["src"], "-i", wav,
         "-filter_complex", vf + ";[2:a]apad[a]", "-map", "[out]", "-map", "[a]", "-t", f"{d:.3f}", *ENC, out])
    return 1 / k, d


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()
    work = args.work or Path(tempfile.mkdtemp(prefix="demo_"))
    work.mkdir(parents=True, exist_ok=True)
    voice = pick_voice()
    print("voice:", voice, "| work dir:", work)

    segs = storyboard()
    audio = {s["name"]: build_audio(s["say"], voice, work, s["name"]) for s in segs}
    pages = {s["name"]: s["html"] for s in segs if s["kind"] == "slide"}
    pages |= {f"{s['name']}_frame": frame_html(s["caption"]) for s in segs if s["kind"] == "clip"}
    render_pngs(pages, work)

    parts, t0, cues, md = [], 0.0, [], ["# Demo video: narration script", "",
                                         f"Voice: macOS text-to-speech, **{voice}** (Indian English, male). "
                                         "Built by [`demo/build/make_demo.py`](build/make_demo.py).", ""]
    for s in segs:
        wav, d, seg_cues = audio[s["name"]]
        out = work / f"{s['name']}.mp4"
        if s["kind"] == "slide":
            build_slide(work / f"{s['name']}.png", wav, d, out)
            label = "slide"
        else:
            speed, d = build_clip(s, work / f"{s['name']}_frame.png", wav, d, out)
            label = f"screen recording `{s['src']}` {s['start']:.0f}–{s['end']:.0f} s at {speed:.2f}× speed"
        parts.append(out)
        cues += [(t0 + a, t0 + b, txt) for a, b, txt in seg_cues]
        md += [f"## {srt_time(t0)[:8]} · {s['name'][3:].replace('_', ' ')} ({label})", "", *[f"> {x}" for x in s["say"]], ""]
        print(f"{s['name']:24s} {d:6.1f}s  {label}")
        t0 += d

    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run([FF, "-y", "-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT_MP4])
    OUT_SRT.write_text("\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1)))
    md.insert(4, f"Length: {int(t0 // 60)} min {t0 % 60:.0f} s · subtitles: [`SupportTicketRouter_demo.srt`](SupportTicketRouter_demo.srt)\n")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n{OUT_MP4}  {t0 / 60:.1f} min  {OUT_MP4.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
