"""Create clearly labeled interface fixtures; these are never research data."""
from pathlib import Path
import json

from attentionseekers.data import NEUTRAL_SYSTEM
from attentionseekers.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASE = {r["scenario"]: r for r in read_jsonl(ROOT / "Data_Creation/data/scenarios.jsonl")}
SCENARIOS = ("code_01", "explain_01", "brainstorm_01")
RESPONSES = {
    "code_01": "Use three pointers: previous, current, and next. Move through the list, reverse each link, and return previous.",
    "explain_01": "A random forest trains many trees on sampled data and combines their predictions. Averaging reduces the variance of one tree.",
    "brainstorm_01": "Consider robot learning, safe navigation, and human-robot collaboration. Each offers practical questions to explore.",
}
SYS = {
    "E": {
        "+": ("You are an outgoing, lively assistant. Speak energetically.",
              "You are a sociable assistant. Use an enthusiastic tone."),
        "-": ("You are a reserved assistant. Speak quietly and concisely.",
              "You are an introverted assistant. Use a restrained tone."),
    },
    "A": {
        "+": ("You are a warm, cooperative assistant. Be considerate.",
              "You are a kind assistant. Use a gentle and accommodating tone."),
        "-": ("You are a blunt, skeptical assistant. Be direct.",
              "You are a tough-minded assistant. Use a terse and challenging tone."),
    },
}
USER_PREFIX = {
    "E": {
        "+": ("Hey, I'd love to explore this together! ", "This is exciting, let's dig in! "),
        "-": ("Briefly, please: ", "I'd prefer a quiet, concise answer: "),
    },
    "A": {
        "+": ("Could you kindly help me with this? ", "I'd appreciate your help, please. "),
        "-": ("Just answer this directly: ", "Skip the niceties and tell me: "),
    },
}


def row(base, *, set_name, trait, sys_pole, user_pole, sys_paraphrase=0,
        user_paraphrase=0, system=NEUTRAL_SYSTEM, user=None):
    scenario = base["scenario"]
    return {"id": f"sample|{scenario}|{set_name}|{trait}|s{sys_pole}{sys_paraphrase}|u{user_pole}{user_paraphrase}",
            "scenario": scenario, "domain": base["domain"], "set": set_name, "trait": trait,
            "sys_pole": sys_pole, "user_pole": user_pole, "sys_paraphrase": sys_paraphrase,
            "user_paraphrase": user_paraphrase, "system": system,
            "user": user if user is not None else base["user"],
            "forced_response": RESPONSES[scenario], "is_sample": True}


def build():
    rows = []
    for scenario in SCENARIOS:
        base = BASE[scenario]
        rows.append(row(base, set_name="neutral", trait="none", sys_pole="0", user_pole="0"))
        for trait in ("E", "A"):
            for pole in ("+", "-"):
                for k, system in enumerate(SYS[trait][pole]):
                    rows.append(row(base, set_name="sys_twin", trait=trait, sys_pole=pole,
                                    user_pole="0", sys_paraphrase=k, system=system))
                for k, prefix in enumerate(USER_PREFIX[trait][pole]):
                    rows.append(row(base, set_name="user_variant", trait=trait, sys_pole="0",
                                    user_pole=pole, user_paraphrase=k, user=prefix + base["user"]))
    return rows


if __name__ == "__main__":
    path = ROOT / "Data_Creation/samples/ws2_sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, build())
    print(f"Wrote {len(build())} labeled sample rows to {path}")
