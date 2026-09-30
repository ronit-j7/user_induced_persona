from pathlib import Path

import numpy as np
import pytest

from attentionseekers.chat import encode
from attentionseekers.config import ModelConfig
from attentionseekers.data import load_rows, matched_pairs, validate_rows
from attentionseekers.heads import bh_qvalues, head_contributions, localize, normalize_scores
from attentionseekers.io import finish_run, load_manifest, write_json
from attentionseekers.repro import encode_upstream, load_upstream_csvs
from attentionseekers.smoke import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Data_Creation/samples/ws2_sample.jsonl"


def test_sample_contract_and_confounded_pair_rejected():
    rows = load_rows(SAMPLE, allow_sample=True)
    assert len(rows) == 51
    assert len(matched_pairs(rows, "E", "assigned")) == 6
    assert len(matched_pairs(rows, "E", "user")) == 6
    tampered = [r.copy() for r in rows]
    target = next(r for r in tampered if r["set"] == "sys_twin" and r["sys_pole"] == "+")
    target["forced_response"] = "different"
    with pytest.raises(ValueError, match="fixed response"):
        validate_rows(tampered, allow_sample=True)
    with pytest.raises(ValueError, match="Sample data"):
        validate_rows(rows)


def test_token_spans_and_response_boundary():
    row = next(r for r in load_rows(SAMPLE, allow_sample=True) if r["set"] == "user_variant")
    tokenizer = CharacterTokenizer()
    encoded = encode(tokenizer, row)
    assert tokenizer.decode(encoded.ids[slice(*encoded.user_span)]) == row["user"]
    assert tokenizer.decode(encoded.ids[slice(*encoded.response_span)]) == row["forced_response"]
    assert tokenizer.decode([encoded.ids[encoded.first]]) == "\n"
    changed = {**row, "forced_response": "A different answer."}
    other = encode(tokenizer, changed)
    assert encoded.ids[:encoded.prefix_length] == other.ids[:other.prefix_length]
    assert encoded.response_span != other.response_span


def test_head_score_identity_and_planted_head():
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(12, 12)).astype(np.float32)
    delta = np.zeros((3, 4), dtype=np.float32)
    delta[1] = np.array([1, 2, 3, 4])
    scores, projected = head_contributions(delta, weight)
    assert np.argmax(scores) == 1
    assert np.isclose(scores.sum(), np.linalg.norm(projected.sum(axis=0)) ** 2, rtol=1e-5)
    raw_z = normalize_scores(np.array([scores]))
    log_z = normalize_scores(np.array([scores]), signed_log=True)
    assert raw_z.shape == (1, 3) and log_z.shape == (1, 3)


def test_clustered_localization_and_bh():
    rows = [r for r in load_rows(SAMPLE, allow_sample=True) if r["trait"] in ("none", "E")]
    n = len(rows)
    acts = np.zeros((n, 2, 4, 8), dtype=np.float32)
    rng = np.random.default_rng(3)
    acts += rng.normal(0, 0.05, size=acts.shape)
    for i, row in enumerate(rows):
        if row["set"] == "sys_twin":
            acts[i, 1, 2, 0] += 5 if row["sys_pole"] == "+" else -5
    weights = [np.eye(32, dtype=np.float32) for _ in range(2)]
    result = localize(acts, rows, weights, "E", "assigned", permutations=31, seed=0)
    assert result["n_pairs"] == 6 and result["n_scenarios"] == 3
    assert np.argmax(result["raw"][1]) == 2
    assert result["raw"].shape == (2, 4)
    assert np.all((result["q"] >= result["p"]) & (result["q"] <= 1))
    assert np.allclose(bh_qvalues(np.array([0.01, 0.04, 0.5])), [0.03, 0.06, 0.5])


def test_tiny_qwen_hooks_response_sensitivity_and_removal():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attentionseekers.extract import capture_encoded
    row = next(r for r in load_rows(SAMPLE, allow_sample=True) if r["set"] == "neutral")
    cfg = ModelConfig(model_name="tiny-test", revision="none", num_layers=2, num_heads=4,
                      num_kv_heads=2, head_dim=8, hidden_size=32, smh_layer=1,
                      smh_heads=(0, 1, 2), dtype="float32", device="cpu", max_length=512)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        pad_token_id=0, eos_token_id=1)).eval()
    tok = CharacterTokenizer()
    first = encode(tok, row)
    second = encode(tok, {**row, "forced_response": "Different forced reply."})
    a, _ = capture_encoded(model, cfg, first)
    b, _ = capture_encoded(model, cfg, second)
    assert a["first"].shape == (2, 4, 8)
    assert np.array_equal(a["first"], b["first"])
    assert np.array_equal(a["user"], b["user"])
    assert not np.array_equal(a["resp"], b["resp"])
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks
    assert not model.model.layers[0].self_attn.o_proj._forward_hooks


def test_upstream_csv_pair_filter_and_tokenization(tmp_path):
    header = "humorous,coherence,prompt,answer\n"
    pos = tmp_path / "pos.csv"
    neg = tmp_path / "neg.csv"
    pos.write_text(header + "80,85,PROMPT+,answer+\n90,80,PROMPT2+,answer2+\n")
    neg.write_text(header + "10,82,PROMPT-,answer-\n60,80,PROMPT2-,answer2-\n")
    rows, details = load_upstream_csvs(pos, neg, "humorous")
    assert details["selected_csv_rows"] == [0]
    assert len(rows) == 2 and rows[0]["pole"] == "+" and rows[1]["pole"] == "-"
    encoded = encode_upstream(CharacterTokenizer(), rows[0])
    assert encoded.user_span is None
    assert encoded.response_span == (len("PROMPT+"), len("PROMPT+answer+"))


def test_manifest_detects_changed_artifact(tmp_path):
    write_json(tmp_path / "data.json", {"value": 1})
    finish_run(tmp_path, {"kind": "test", "is_sample": True})
    assert load_manifest(tmp_path)["status"] == "complete"
    write_json(tmp_path / "data.json", {"value": 2})
    with pytest.raises(ValueError, match="changed"):
        load_manifest(tmp_path)
