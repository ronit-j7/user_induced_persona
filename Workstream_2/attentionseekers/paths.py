"""Repository locations for the consolidated Workstream 2 implementation."""
from pathlib import Path

WORKSTREAM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSTREAM_ROOT.parent
DEFAULT_CONFIG = WORKSTREAM_ROOT / "configs/qwen.json"
SAMPLE_DATA = WORKSTREAM_ROOT / "samples/ws2_sample.jsonl"
