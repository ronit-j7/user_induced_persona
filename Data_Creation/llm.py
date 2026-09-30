"""API clients for the rewriter (DeepSeek) and the QC judge (OpenAI), with an on-disk cache and usage tracking.

Every successful call is cached by sha256 of its full request, so re-running the pipeline never pays twice
and a crashed run resumes almost for free.
"""

import hashlib
import json
import os
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache" / "llm"

# USD per 1M tokens. DeepSeek uses peak prices (off-peak is half), so estimates are conservative.
PRICES = {
    "deepseek-flash": {"input": 0.30, "cached_input": 0.006, "output": 1.20},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
}

PARSE_RETRIES = 3


class LLMError(Exception):
    pass


def parse_json(text):
    """Parse a JSON object from model output, tolerating code fences and surrounding text."""
    if not text:
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _get(obj, *path, default=0):
    """Read a nested field from an SDK object or dict, including provider-specific extra fields."""
    for name in path:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(name)
        else:
            val = getattr(obj, name, None)
            if val is None and getattr(obj, "model_extra", None):
                val = obj.model_extra.get(name)
            obj = val
    return default if obj is None else obj


class DiskCache:
    def __init__(self, directory=CACHE_DIR, enabled=True):
        self.dir = Path(directory)
        self.enabled = enabled
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, payload):
        if not self.enabled:
            return None
        path = self.dir / f"{self.key(payload)}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def put(self, payload, value):
        if not self.enabled:
            return
        path = self.dir / f"{self.key(payload)}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": payload, "response": value}, ensure_ascii=False))
        tmp.replace(path)


class Usage:
    """Thread-safe token and cost accounting, per model."""

    FIELDS = ("calls", "cache_hits_local", "input_tokens", "cached_input_tokens", "output_tokens",
              "reasoning_tokens", "failed_parses")

    def __init__(self):
        self.lock = threading.Lock()
        self.by_model = {}

    def add(self, model, **counts):
        with self.lock:
            row = self.by_model.setdefault(model, {f: 0 for f in self.FIELDS})
            for k, v in counts.items():
                row[k] += v or 0

    def summary(self):
        out = {}
        for model, row in self.by_model.items():
            price = PRICES.get(model)
            cost = None
            if price:
                uncached = row["input_tokens"] - row["cached_input_tokens"]
                cost = (uncached * price["input"] + row["cached_input_tokens"] * price["cached_input"]
                        + row["output_tokens"] * price["output"]) / 1e6
            out[model] = dict(row, cost_usd=None if cost is None else round(cost, 4))
        return out


def _openai_key():
    for name in ("GPT_API_KEY", "GPT_APT_KEY", "OPENAI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise LLMError("No OpenAI key found. Export GPT_API_KEY (or GPT_APT_KEY / OPENAI_API_KEY).")


class DeepSeekRewriter:
    """Rewriter on DeepSeek's OpenAI-compatible Chat Completions API."""

    def __init__(self, cfg, usage, cache):
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMError("DEEPSEEK_API_KEY is not set.")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com", max_retries=5, timeout=300)
        self.cfg, self.usage, self.cache = cfg, usage, cache

    def complete_json(self, messages, validate):
        cfg = self.cfg
        kwargs = {"model": cfg["model"], "messages": messages, "max_tokens": cfg["max_tokens"],
                  "extra_body": {"thinking": {"type": "enabled" if cfg["thinking"] else "disabled"}}}
        if cfg["thinking"]:
            kwargs["reasoning_effort"] = cfg["reasoning_effort"]
        else:
            kwargs["temperature"] = cfg["temperature"]
        cached = self.cache.get(kwargs)
        if cached is not None:
            self.usage.add(cfg["model"], cache_hits_local=1)
            return cached["response"]
        last_err = None
        for _ in range(PARSE_RETRIES):
            resp = self.client.chat.completions.create(**kwargs)
            u = resp.usage
            self.usage.add(cfg["model"], calls=1,
                           input_tokens=_get(u, "prompt_tokens"),
                           cached_input_tokens=_get(u, "prompt_cache_hit_tokens"),
                           output_tokens=_get(u, "completion_tokens"),
                           reasoning_tokens=_get(u, "completion_tokens_details", "reasoning_tokens"))
            try:
                obj = parse_json(resp.choices[0].message.content)
                validate(obj)
            except (ValueError, KeyError, TypeError) as e:
                last_err = e
                self.usage.add(cfg["model"], failed_parses=1)
                continue
            self.cache.put(kwargs, obj)
            return obj
        raise LLMError(f"DeepSeek returned unusable output {PARSE_RETRIES} times: {last_err}")

    def rewrite(self, messages, n):
        def validate(obj):
            rw = obj["rewrites"]
            if not (isinstance(rw, list) and len(rw) == n and all(isinstance(x, str) for x in rw)):
                raise ValueError(f"expected {n} rewrites")
        return [s.strip() for s in self.complete_json(messages, validate)["rewrites"]]


class OpenAIJudge:
    """QC judge on the OpenAI Responses API with strict Structured Outputs."""

    def __init__(self, cfg, usage, cache):
        from openai import OpenAI
        self.client = OpenAI(api_key=_openai_key(), max_retries=5, timeout=300)
        self.cfg, self.usage, self.cache = cfg, usage, cache

    def complete_schema(self, messages, schema, name, effort):
        kwargs = {"model": self.cfg["model"], "input": messages, "reasoning": {"effort": effort},
                  "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}}
        cached = self.cache.get(kwargs)
        if cached is not None:
            self.usage.add(self.cfg["model"], cache_hits_local=1)
            return cached["response"]
        last_err = None
        for _ in range(PARSE_RETRIES):
            resp = self.client.responses.create(**kwargs)
            u = resp.usage
            self.usage.add(self.cfg["model"], calls=1,
                           input_tokens=_get(u, "input_tokens"),
                           cached_input_tokens=_get(u, "input_tokens_details", "cached_tokens"),
                           output_tokens=_get(u, "output_tokens"),
                           reasoning_tokens=_get(u, "output_tokens_details", "reasoning_tokens"))
            try:
                obj = json.loads(resp.output_text)
            except (ValueError, TypeError) as e:
                last_err = e
                self.usage.add(self.cfg["model"], failed_parses=1)
                continue
            self.cache.put(kwargs, obj)
            return obj
        raise LLMError(f"Judge returned unusable output {PARSE_RETRIES} times: {last_err}")

    def checklist(self, messages, schema):
        return self.complete_schema(messages, schema, "qc_checklist", self.cfg["checklist_effort"])

    def forced_choice(self, messages, schema):
        return self.complete_schema(messages, schema, "qc_forced_choice",
                                    self.cfg["forced_choice_effort"])["choice"]
