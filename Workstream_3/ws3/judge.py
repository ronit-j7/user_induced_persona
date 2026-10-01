"""Logprob-weighted 0-100 judge. Cached by sha256 of the request. Never sends the user message."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from .rubrics import METRICS, render

SECRETS = Path("/media/gaurav/Data21/eshaan/secrets/.env")


def score_logprobs(token_probs, *, min_mass=0.25):
    """Probability-weighted mean over integer tokens 0-100. None if mass < min_mass."""
    total = 0.0
    weighted = 0.0
    for key, value in token_probs.items():
        try:
            number = int(str(key).strip())
        except ValueError:
            continue
        if 0 <= number <= 100:
            total += float(value)
            weighted += number * float(value)
    if total < min_mass:
        return None
    return weighted / total


def load_api_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_BASE_URL")
    if SECRETS.is_file():
        values = {}
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        if values.get("OPENAI_API_KEY"):
            return values["OPENAI_API_KEY"], values.get("OPENAI_BASE_URL")
    return None, None


def _cache_path(cache_dir, payload):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def call_logprobs(client, model, prompt, *, top_logprobs):
    import math
    last_error = None
    completion = None
    for attempt in range(6):
        succeeded = False
        retry_later = False
        for token_arg in ({"max_tokens": 1}, {"max_completion_tokens": 1}):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    seed=0,
                    **token_arg,
                )
                succeeded = True
                break
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "max_tokens" in message or "max_completion_tokens" in message:
                    continue
                if "429" in message or "rate limit" in message or "timeout" in message or "500" in message:
                    time.sleep(min(30, 2 ** attempt))
                    retry_later = True
                    break
                raise
        if succeeded:
            break
        if not retry_later:
            break
    if completion is None:
        raise last_error
    choice = completion.choices[0]
    content = getattr(choice, "logprobs", None)
    items = []
    if content and getattr(content, "content", None):
        items = content.content[0].top_logprobs or []
    probs = {token: math.exp(value) for token, value in
             ((item.token, float(item.logprob)) for item in items)}
    usage = getattr(completion, "usage", None)
    tokens = {
        "input": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    return probs, tokens


def judge_one(client, model, metric, intent, answer, *, cache_dir, min_mass, top_logprobs, forbidden=()):
    prompt = render(metric, intent=intent, answer=answer, forbidden=forbidden)
    payload = {"model": model, "metric": metric, "prompt": prompt, "top_logprobs": top_logprobs}
    path = _cache_path(cache_dir, payload)
    if path.is_file():
        cached = json.loads(path.read_text())
        return cached["score"], cached["tokens"], True
    probs, tokens = call_logprobs(client, model, prompt, top_logprobs=top_logprobs)
    score = score_logprobs(probs, min_mass=min_mass)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"request": payload, "score": score, "tokens": tokens, "probs": probs}))
    tmp.replace(path)
    return score, tokens, False


def judge_file(records, out, *, model, cache_dir, min_mass=0.25, top_logprobs=20,
               fallback_model=None, base_url=None, workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    key, env_base = load_api_key()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set and secrets/.env is missing")
    client = OpenAI(api_key=key, base_url=base_url or env_base)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "scores.jsonl"
    done = {}
    if done_path.is_file():
        for line in done_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    pending = [row for row in records if row["id"] not in done]
    usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
    started = time.time()
    tasks = []
    for record in pending:
        if "user" in record:
            raise ValueError("Refusing to judge a record that still carries the user message")
        for metric in METRICS:
            tasks.append((record, metric))

    def work(item):
        record, metric = item
        # Stay on one judge model. call_logprobs retries parameter and rate-limit errors.
        score, tokens, hit = judge_one(
            client, model, metric, record["intent"], record["text"],
            cache_dir=cache_dir, min_mass=min_mass, top_logprobs=top_logprobs)
        return record["id"], metric, score, tokens, hit, model

    results = {}
    finished = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, item) for item in tasks]
        for future in as_completed(futures):
            row_id, metric, score, tokens, hit, chosen = future.result()
            with lock:
                results.setdefault(row_id, {})[metric] = {"score": score, "model": chosen}
                usage["calls"] += 0 if hit else 1
                usage["cache_hits"] += 1 if hit else 0
                usage["input_tokens"] += 0 if hit else tokens["input"]
                usage["output_tokens"] += 0 if hit else tokens["output"]
                finished += 1
                complete = all(name in results[row_id] for name in METRICS)
                if finished % 20 == 0 or finished == len(tasks):
                    elapsed = time.time() - started
                    rate = finished / max(elapsed, 1e-6)
                    eta = (len(tasks) - finished) / max(rate, 1e-6)
                    print(f"[progress] {finished}/{len(tasks)} eta_s={eta:.0f}", flush=True)
            if not complete:
                continue
    by_id = {row["id"]: row for row in records}
    with open(done_path, "a", encoding="utf-8") as handle:
        for row_id, metrics in results.items():
            if any(name not in metrics for name in METRICS):
                continue
            source = by_id[row_id]
            refusal = metrics["refusal"]["score"]
            saved = {
                "id": row_id,
                "scenario": source["scenario"],
                "domain": source["domain"],
                "trait": source["trait"],
                "pole": source["pole"],
                "k": source["k"],
                "sample": source["sample"],
                "kind": source["kind"],
                "n_tokens": source["n_tokens"],
                "E": metrics["E"]["score"],
                "A": metrics["A"]["score"],
                "coherence": metrics["coherence"]["score"],
                "refusal": refusal,
                "refused": None if refusal is None else bool(refusal >= 50),
            }
            handle.write(json.dumps(saved) + "\n")
            done[row_id] = saved
    (out / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    return done_path


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Judge generations without seeing the user message")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--ws3-config", default="Workstream_3/configs/ws3.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    settings = json.loads(Path(args.ws3_config).read_text())
    key, _base = load_api_key()
    if not key:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        status = {"status": "BLOCKED", "reason": "OPENAI_API_KEY missing"}
        (Path(args.out) / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        return
    rows = []
    for line in Path(args.generations).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    judge_file(
        rows, args.out, model=settings["judge_model"], cache_dir=args.cache,
        min_mass=settings["min_integer_mass"], top_logprobs=settings["top_logprobs"],
        fallback_model=settings["judge_fallback_model"], base_url=settings.get("judge_base_url"),
        workers=args.workers)


if __name__ == "__main__":
    main()
