"""Shared constants for the support-ticket router (kept identical to the course notebook)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_FACTORY_DIR = ROOT / "LLaMA-Factory"
RAW_CSV = ROOT / "data" / "raw" / "support_tickets.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
DATASET_NAME = "support_tickets"   # name registered in LLaMA-Factory's dataset_info.json
SEED = 42
VAL_SIZE = 0.2

# Label order used throughout the course notebook (also the A-G order of the baseline prompt).
LABEL2ID = {
    "Support general": 0,
    "Fileservice": 1,
    "O365": 2,
    "EOL": 3,
    "Software": 4,
    "Active Directory": 5,
    "Computer-Services": 6,
}
LABELS = list(LABEL2ID)

SYSTEM_PROMPT = (
    "You are an IT helpdesk ticket routing assistant. "
    "Given a support ticket, respond with exactly one of the following categories: "
    "Support general, Fileservice, O365, EOL, Software, Active Directory, Computer-Services."
)


def format_user(ticket_text: str) -> str:
    """User turn exactly as used in training and inference."""
    return f"Support Ticket: {ticket_text}"
