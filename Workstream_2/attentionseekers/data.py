"""Version 1 WS1 -> extraction interface, with no dependency on a model runtime."""
from collections import Counter, defaultdict
import math
import re

from .io import read_jsonl

NEUTRAL_SYSTEM = "You are a helpful assistant."
SETS = {"sys_twin", "user_variant", "neutral", "factorial"}
REQUIRED = {"id", "scenario", "domain", "set", "trait", "sys_pole", "user_pole",
            "sys_paraphrase", "user_paraphrase", "system", "user", "forced_response"}


def qc_protocol(row):
    """Validate the actual WS1 QC export without fabricating numeric judge scores."""
    name, qc = row["id"], row.get("qc", {})
    if not isinstance(qc, dict):
        raise ValueError(f"{name}: qc must be an object")
    if qc.get("protocol") == "authored_system_prompt":
        if row["set"] != "sys_twin" or not re.fullmatch(r"[0-9a-f]{64}", str(qc.get("source_sha256", ""))):
            raise ValueError(f"{name}: authored_system_prompt requires a sys_twin and source SHA256")
        return "authored_system_prompt"
    if qc.get("passed") is not True:
        raise ValueError(f"{name}: real rows require qc.passed=true")
    if "forced_choice" in qc:
        if row["set"] not in {"user_variant", "factorial"}:
            raise ValueError(f"{name}: forced-choice QC applies to user rewrites")
        fc = qc["forced_choice"]
        # D4: a high pole passes when the rewrite wins both orders, a low pole when the original does.
        expected = {"+": ("B", "A"), "-": ("A", "B")}.get(row.get("user_pole"))
        if expected is None or not isinstance(fc, dict) or (fc.get("rewrite_as_B"), fc.get("rewrite_as_A")) != expected:
            raise ValueError(f"{name}: forced-choice QC must match the pole in both orders")
        if "checklist" in qc:
            checklist = qc["checklist"]
            checks = ("C1", "C2", "C3", "C4", "L1", "L2", "R1", "R2", "R3")
            if not isinstance(checklist, dict) or any(
                not isinstance(checklist.get(k), dict) or checklist[k].get("answer") != "yes" for k in checks
            ):
                raise ValueError(f"{name}: checklist QC requires all nine checks to pass")
        return "ws1_checklist_forced_choice"
    if row["set"] == "factorial" and isinstance(qc.get("user_variant_id"), str) and qc["user_variant_id"].strip():
        return "factorial_user_variant_link"
    for key in ("content", "trait"):
        score = qc.get(key)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 70 <= score <= 100:
            raise ValueError(f"{name}: qc.{key} must be a passing score in [70,100]")
    return "numeric_scores"


def validate_rows(rows, *, allow_sample=False, require_response=True):
    if not rows:
        raise ValueError("Dataset is empty")
    ids, responses, neutral_users, neutral_systems = set(), {}, {}, set()
    domains, sample_flags, qc_counts = {}, set(), Counter()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError(f"row {i}: missing fields {sorted(REQUIRED - row.keys()) if isinstance(row, dict) else REQUIRED}")
        name = row["id"]
        for field in ("id", "scenario", "domain", "trait", "system", "user"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{name}: {field} must be a nonempty string")
        if name in ids:
            raise ValueError(f"Duplicate id: {name}")
        ids.add(name)
        if row["set"] not in SETS or any(row[k] not in {"+", "-", "0"} for k in ("sys_pole", "user_pole")):
            raise ValueError(f"{name}: invalid set or pole")
        for key in ("sys_paraphrase", "user_paraphrase"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"{name}: {key} must be a nonnegative integer")
        sample = row.get("is_sample", False)
        if type(sample) is not bool:
            raise ValueError(f"{name}: is_sample must be boolean")
        sample_flags.add(sample)
        if sample and not allow_sample:
            raise ValueError("Sample data requires explicit --allow-sample")
        if not sample and row["set"] != "neutral":
            qc_counts[qc_protocol(row)] += 1
        scenario = row["scenario"]
        if scenario in domains and domains[scenario] != row["domain"]:
            raise ValueError(f"{name}: inconsistent scenario domain")
        domains[scenario] = row["domain"]
        response = row["forced_response"]
        if response is not None and (not isinstance(response, str) or not response.strip()):
            raise ValueError(f"{name}: forced_response must be nonempty text or null")
        if require_response and response is None:
            raise ValueError(f"{name}: forced_response required for resp extraction")
        if response is not None:
            if scenario in responses and responses[scenario] != response:
                raise ValueError(f"{name}: fixed response differs within scenario")
            responses[scenario] = response
        s, u = row["sys_pole"], row["user_pole"]
        kind = row["set"]
        if (kind == "neutral" and (row["trait"] != "none" or row["sys_paraphrase"] != 0 or row["user_paraphrase"] != 0)) or (kind != "neutral" and row["trait"] not in {"E", "A"}):
            raise ValueError(f"{name}: invalid trait or neutral paraphrase")
        if kind == "sys_twin" and (s not in {"+", "-"} or u != "0" or row["user_paraphrase"] != 0):
            raise ValueError(f"{name}: sys_twin needs sys +/- and neutral user with paraphrase 0")
        if kind == "user_variant" and (u not in {"+", "-"} or s != "0" or row["sys_paraphrase"] != 0):
            raise ValueError(f"{name}: user_variant needs user +/- and neutral system with paraphrase 0")
        if kind == "neutral" and (s != "0" or u != "0"):
            raise ValueError(f"{name}: neutral rows need both poles 0")
        if kind == "factorial" and u not in {"+", "-"}:
            raise ValueError(f"{name}: factorial user pole must be +/-")
        if u == "0":
            if scenario in neutral_users and neutral_users[scenario] != row["user"]:
                raise ValueError(f"{name}: neutral user differs within scenario")
            neutral_users[scenario] = row["user"]
        if s == "0":
            neutral_systems.add(row["system"])
    if len(sample_flags) != 1:
        raise ValueError("Cannot mix sample and real rows")
    if neutral_systems and neutral_systems != {NEUTRAL_SYSTEM}:
        raise ValueError(f"Neutral system must be exactly {NEUTRAL_SYSTEM!r}")
    counts = Counter((r["set"], r["trait"], r["sys_pole"], r["user_pole"]) for r in rows)
    return {"rows": len(rows), "scenarios": len(domains), "is_sample": sample_flags == {True},
            "qc_protocol_counts": dict(qc_counts),
            "cells": [{"set": k[0], "trait": k[1], "sys_pole": k[2], "user_pole": k[3], "n": v}
                      for k, v in sorted(counts.items())]}


def load_rows(path, **kwargs):
    rows = read_jsonl(path)
    validate_rows(rows, **kwargs)
    return rows


def matched_pairs(rows, trait, source):
    """Return (plus index, minus index, scenario) pairs; never silently drop rows."""
    if source not in {"assigned", "user"}:
        raise ValueError("source must be assigned or user")
    kind, pole = ("sys_twin", "sys_pole") if source == "assigned" else ("user_variant", "user_pole")
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        if row["set"] != kind or row["trait"] != trait:
            continue
        key = (row["scenario"], row["user_paraphrase"], row["sys_paraphrase"])
        if row[pole] in groups[key]:
            raise ValueError(f"Duplicate pole for pair {key}")
        groups[key][row[pole]] = i
    if not groups:
        raise ValueError(f"No {source} rows for trait {trait}")
    output, cells = [], defaultdict(set)
    for key, poles in sorted(groups.items()):
        if set(poles) != {"+", "-"}:
            raise ValueError(f"Unmatched poles at {key}")
        a, b = rows[poles["+"]], rows[poles["-"]]
        fixed = "user" if source == "assigned" else "system"
        if a[fixed] != b[fixed] or a["forced_response"] != b["forced_response"]:
            raise ValueError(f"Confounded contrast at {key}")
        if a["system" if source == "assigned" else "user"] == b["system" if source == "assigned" else "user"]:
            raise ValueError(f"Identical styled text at {key}")
        cells[key[0]].add(key[1:])
        output.append((poles["+"], poles["-"], key[0]))
    if len({frozenset(v) for v in cells.values()}) != 1:
        raise ValueError("Unbalanced paraphrase cells across scenarios")
    return output
