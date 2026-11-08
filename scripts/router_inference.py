"""Inference helpers shared by Phases 5 and 6, a faithful port of the course notebook's cells.

* ``classify_base``: zero-shot **baseline**. The base model sees the seven labels as options A–G,
  and we compare the logits of the seven letter tokens at the first generated position (the
  course's constrained multiple-choice format, which gives the untrained model its best shot).
* ``classify``: the **fine-tuned router**. The training system prompt + "Support Ticket: …",
  greedy generation of up to 10 tokens, prefix-match against the label set (fallback
  "Support general"), and a confidence = softmax over the seven labels' first tokens at step 1.

Both use the tokenizer **saved next to the adapter**. LLaMA-Factory rewrote its chat template to
the `default` training format (``System: …<|endoftext|>\\nHuman: …<|endoftext|>\\nAssistant:``),
so prompts at inference are byte-identical to training.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from router_common import ROOT, SYSTEM_PROMPT, format_user

ADAPTER_DIR = ROOT / "adapters" / "qwen3-1.7b-support-router-lora"
# ".nosync" keeps the 3.4 GB merged model out of iCloud Drive (this repo lives in ~/Documents).
MERGED_DIR = ROOT / "models.nosync" / "qwen3-1.7b-support-router-merged"

# Course order (also LABEL2ID order); index i <-> letter "ABCDEFG"[i]
LABEL_TOKENS = ["Support general", "Fileservice", "O365", "EOL", "Software", "Active Directory", "Computer-Services"]
CHOICES = "ABCDEFG"
CHOICE2LABEL = dict(zip(CHOICES, LABEL_TOKENS))
BASE_SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Classify the ticket by responding with ONLY a single letter — nothing else:\n"
    + "\n".join(f"{ch}) {lbl}" for ch, lbl in CHOICE2LABEL.items())
)

# The course's five smoke tickets (5 of the 7 classes, unambiguous phrasing)
SMOKE_TICKETS = [
    ("Please create a new user account for the new employee starting Monday.", "Active Directory"),
    ("I cannot access the shared network folder — I keep getting Access Denied.", "Fileservice"),
    ("Outlook keeps asking me to re-enter my password and Teams won't sync.", "O365"),
    ("We're ready to retire [SERVER] — please remove it from monitoring.", "EOL"),
    ("Adobe Acrobat fails to install, error code 1603.", "Software"),
]
# Two extra tickets for the classes the course's smoke test leaves out (written in the dataset's style)
EXTRA_SMOKE_TICKETS = [
    ("Hi, the printer on the 2nd floor needs a new driver installed on my machine. Could someone set it up?",
     "Computer-Services"),
    ("Could you please create a Light (external) user account for a contractor? They only need basic email access.",
     "Support general"),
]


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@dataclass
class Prediction:
    label: str
    confidence: float
    raw: str
    latency_ms: float


class BaselineRouter:
    """Zero-shot A–G letter choice on the *untrained* base model (course cell 'Measure Baseline')."""

    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
        self.choice_ids = []
        for ch in CHOICES:
            ids_plain = tokenizer.encode(ch, add_special_tokens=False)
            ids_spaced = tokenizer.encode(f" {ch}", add_special_tokens=False)
            self.choice_ids.append(ids_spaced[0] if len(ids_spaced) == 1 else ids_plain[0])

    def prompt(self, ticket_text: str) -> str:
        messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": format_user(ticket_text)}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, ticket_text: str) -> tuple[str, list[float]]:
        inputs = self.tok(self.prompt(ticket_text), return_tensors="pt").to(self.model.device)
        logits = self.model(input_ids=inputs["input_ids"]).logits[0, -1, :]
        probs = torch.softmax(logits[self.choice_ids].float(), dim=-1).cpu().tolist()
        return CHOICE2LABEL[CHOICES[probs.index(max(probs))]], probs


class Router:
    """The fine-tuned router (course cell 'Define classify()')."""

    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
        self.device = str(model.device.type)
        self.label_first_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LABEL_TOKENS]
        assert len(set(self.label_first_ids)) == len(LABEL_TOKENS), "label first tokens must be distinct"

    def prompt(self, ticket_text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_user(ticket_text)}]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, ticket_text: str, compute_confidence: bool = True) -> Prediction:
        sync(self.device)
        t0 = time.perf_counter()
        inputs = self.tok(self.prompt(ticket_text), return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=self.tok.eos_token_id,
        )
        generated = self.tok.decode(out.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - t0) * 1000  # decode() needs the tokens on CPU, so the GPU work is done

        matched = next((l for l in LABEL_TOKENS if generated.lower().startswith(l.lower())), None)
        if matched is None:
            matched = "Support general"
        conf = 1.0
        if compute_confidence and out.scores:
            first = out.scores[0][0]
            label_probs = torch.softmax(first[self.label_first_ids].float(), dim=-1).cpu().tolist()
            conf = label_probs[LABEL_TOKENS.index(matched)]
        return Prediction(matched, conf, generated, latency_ms)


def load_base(device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from router_common import BASE_MODEL
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map=device).eval()
    tok_src = ADAPTER_DIR if (ADAPTER_DIR / "tokenizer.json").exists() else BASE_MODEL
    return model, AutoTokenizer.from_pretrained(str(tok_src))


def load_merged(device: str, dtype: torch.dtype, path: Path = MERGED_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, device_map=device).eval()
    return model, AutoTokenizer.from_pretrained(str(path))
