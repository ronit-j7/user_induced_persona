"""Regression tests for the current WS1 export and WS2 research-critical gates."""
import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from attentionseekers.chat import describe_encoding, encode
from attentionseekers.config import ModelConfig
from attentionseekers.data import matched_pairs, validate_rows
from attentionseekers.heads import head_contributions, normalize_scores, sign_flip_pvalues
from attentionseekers.io import read_jsonl
from attentionseekers.pipeline import check_run_data
from attentionseekers.prepare import prepare_rows
from attentionseekers.smoke import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[2]
WS2_ROOT = ROOT / "Workstream_2"


@pytest.fixture
def ws1():
    return (read_jsonl(ROOT / "Data_Creation/data/scenarios.jsonl"),
            json.loads((ROOT / "Data_Creation/prompts/sys_prompts.json").read_text()))


def test_actual_ws1_assigned_rows_and_canonical_responses(ws1):
    scenarios, prompts = ws1
    rows, report = prepare_rows(scenarios, prompts)
    assert report["rows"] == 420 and report["rows_per_set"] == {"neutral": 20, "sys_twin": 400}
    assert report["user_data_status"] == "NOT_AVAILABLE"
    expected = {r["scenario"]: r["forced_response"] for r in scenarios}
    assert all(r["forced_response"] == expected[r["scenario"]] for r in rows)
    assert all("content" not in r.get("qc", {}) for r in rows)
    for trait in ("E", "A"):
        assert len(matched_pairs(rows, trait, "assigned")) == 100
    check_run_data(rows)
    with pytest.raises(ValueError, match="not available"):
        check_run_data(rows, require_user=True)


def test_selective_refresh_rejects_other_inputs_and_reuses_neutral(ws1):
    from Workstream_2.scripts.refresh_assigned import changed_rows
    rows, _ = prepare_rows(*ws1)
    changed = [{**r, "system": r["system"] + " Still help fully."}
               if r["set"] == "sys_twin" and r["trait"] == "A" else dict(r) for r in rows]
    selected = changed_rows(rows, changed, "A")
    assert len(selected) == 200
    assert validate_rows(selected)["rows"] == 200
    with pytest.raises(ValueError, match="Neutral rows required"):
        check_run_data(selected, traits=("A",))
    e = next(r for r in changed if r["trait"] == "E")
    e["system"] += " Unexpected edit."
    with pytest.raises(ValueError, match="outside A"):
        changed_rows(rows, changed, "A")
    e["system"] = next(r["system"] for r in rows if r["id"] == e["id"])
    selected[0]["forced_response"] += " Unexpected response edit."
    with pytest.raises(ValueError, match="forced_response"):
        changed_rows(rows, changed, "A")


def test_ws1_make_row_qc_without_invented_numeric_scores(ws1, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "Data_Creation"))
    generator = importlib.import_module("gen_user_variants")
    scn = ws1[0][0]
    accepted = {"text": "Wow, " + scn["user"], "attempts": 1,
                "eval": {"forced_choice": {"rewrite_as_B": "B", "rewrite_as_A": "A"}}}
    row = generator.make_row(scn, "E", "+", 0, accepted, lambda text: None)
    assert validate_rows([row])["qc_protocol_counts"] == {"ws1_checklist_forced_choice": 1}
    assert "content" not in row["qc"] and "trait" not in row["qc"]
    row["qc"]["forced_choice"]["rewrite_as_A"] = "B"
    with pytest.raises(ValueError, match="both orders"):
        validate_rows([row])
    row["qc"]["forced_choice"]["rewrite_as_A"] = "A"
    row["qc"]["checklist"] = {k: {"answer": "yes"} for k in ("C1", "C2", "C3", "C4", "L1", "L2", "R1", "R2", "R3")}
    row["qc"]["checklist"]["C3"]["answer"] = "no"
    with pytest.raises(ValueError, match="nine checks"):
        validate_rows([row])


def test_prepare_rejects_missing_responses_and_unpaired_system_prompts(ws1):
    scenarios, prompts = ws1
    broken = [{**r, "forced_response": None} for r in scenarios]
    with pytest.raises(ValueError, match="forced_response"):
        prepare_rows(broken, prompts)
    prompts = {**prompts, "prompts": prompts["prompts"][1:]}
    with pytest.raises(ValueError, match="Unmatched system prompts"):
        prepare_rows(scenarios, prompts)


def test_prepare_rejects_dropped_user_pairs_and_changed_response(ws1):
    scenarios, prompts = ws1
    rows, _ = prepare_rows(scenarios, prompts, traits=("E",))
    scn = scenarios[0]
    variant = {**next(r for r in rows if r["scenario"] == scn["scenario"] and r["set"] == "neutral"),
               "id": "fixture-user", "set": "user_variant", "trait": "E", "user_pole": "+",
               "user": "Wow, " + scn["user"],
               "qc": {"passed": True, "forced_choice": {"rewrite_as_B": "B", "rewrite_as_A": "A"}}}
    with pytest.raises(ValueError, match="Unmatched poles"):
        prepare_rows(scenarios, prompts, [variant], traits=("E",))
    with pytest.raises(ValueError, match="canonical"):
        prepare_rows(scenarios, prompts, [{**variant, "forced_response": "different"}], traits=("E",))


def test_response_window_preserves_full_tokens_and_records_short_responses(ws1):
    rows, _ = prepare_rows(*ws1)
    row = next(r for r in rows if r["scenario"] == "code_01" and r["set"] == "neutral")
    tok = CharacterTokenizer()
    full, window = encode(tok, row, max_length=10000), encode(tok, row, max_length=10000, response_tokens=150)
    assert full.ids == window.ids
    assert tok.decode(window.ids[slice(*window.response_span)]) == row["forced_response"][:150]
    detail = describe_encoding(tok, window)
    assert detail["response_tokens_used"] == 150
    assert detail["response_tokens_total"] == len(row["forced_response"])
    short = encode(tok, {**row, "forced_response": "Short."}, response_tokens=150)
    assert describe_encoding(tok, short)["response_tokens_used"] == 6
    with pytest.raises(ValueError, match="response_tokens"):
        encode(tok, row, response_tokens=0)


def test_causal_response_window_and_cleanup_after_hook_failure():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attentionseekers.extract import capture_encoded, capture_hooks
    cfg = ModelConfig(num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32,
                      smh_layer=1, smh_heads=(0, 1, 2), dtype="float32", device="cpu", is_synthetic=True)
    torch.manual_seed(2)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)).eval()
    row = {"id": "window", "system": "You are helpful.", "user": "Hello?",
           "forced_response": "This response is deliberately longer than the selected window."}
    encoded = encode(CharacterTokenizer(), row, response_tokens=10)
    trimmed = replace(encoded, ids=encoded.ids[:encoded.response_span[1]])
    a, _ = capture_encoded(model, cfg, encoded)
    b, _ = capture_encoded(model, cfg, trimmed)
    np.testing.assert_allclose(a["resp"], b["resp"], atol=2e-6, rtol=2e-5)
    assert np.array_equal(a["first"], b["first"]) and np.array_equal(a["user"], b["user"])
    with pytest.raises(ValueError, match="Invalid"):
        with capture_hooks(model, cfg, {"resp": (0, 9999)}):
            model.model(torch.tensor([encoded.ids]))
    assert all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers)


def test_scores_match_the_vendored_upstream_implementation():
    torch = pytest.importorskip("torch")
    source = WS2_ROOT / "reference_repos/style-modulation-head/src/head_analysis/head_contribution/compute.py"
    names = {"inner_product_similarity", "compute_head_contributions", "normalize_matrix"}
    definitions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    rng = np.random.default_rng(17)
    delta, weight, target = (rng.normal(size=shape).astype(np.float32) for shape in ((4, 8), (32, 32), (32,)))
    ours, _ = head_contributions(delta, weight, target)
    upstream = namespace["compute_head_contributions"](torch.from_numpy(delta.ravel()),
                                                        torch.from_numpy(target), torch.from_numpy(weight), 4, 8)
    np.testing.assert_allclose(ours, upstream, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(normalize_scores(ours[None]), namespace["normalize_matrix"](ours[None]))


def test_permutation_optimization_matches_direct_head_scores():
    rng = np.random.default_rng(23)
    diffs = rng.normal(size=(4, 2, 3, 4)).astype(np.float32)
    weights = [rng.normal(size=(12, 12)).astype(np.float32) for _ in range(2)]
    observed = np.stack([head_contributions(diffs.mean(0)[l], weights[l])[0] for l in range(2)])
    optimized = sign_flip_pvalues(diffs, weights, observed, permutations=31, seed=5)
    signs = np.random.default_rng(5).choice((-1., 1.), size=(31, 4)).astype(np.float32)
    count = np.zeros((2, 3), dtype=int)
    for sign in signs:
        delta = (diffs * sign[:, None, None, None]).mean(0)
        null = np.stack([head_contributions(delta[l], weights[l])[0] for l in range(2)])
        count += null >= observed - 1e-7
    np.testing.assert_array_equal(optimized, (count + 1) / 32)


def tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = ModelConfig(model_name="tiny-integration", num_layers=2, num_heads=4, num_kv_heads=2,
                         head_dim=8, hidden_size=32, smh_layer=1, smh_heads=(0, 1, 2),
                         dtype="float32", device="cpu", response_tokens=150, is_synthetic=True)
    torch.manual_seed(13)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)).eval()
    return model, config


def test_independent_vendored_extraction_hooks_without_judge_credentials():
    torch = pytest.importorskip("torch")
    from transformers import BatchEncoding
    from attentionseekers.reference_check import verify_reference_extraction

    class TensorTokenizer(CharacterTokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
            value = super().__call__(text, add_special_tokens, return_offsets_mapping)
            if return_tensors:
                value = {"input_ids": torch.tensor([value["input_ids"]]),
                         "attention_mask": torch.ones((1, len(value["input_ids"])), dtype=torch.long)}
            return BatchEncoding(value)

    model, config = tiny_model()
    rows = read_jsonl(WS2_ROOT / "samples/ws2_sample.jsonl")
    a, b, _ = matched_pairs(rows, "E", "assigned")[0]
    report = verify_reference_extraction(model, TensorTokenizer(), config, [rows[a], rows[b]])
    assert report["passed"] and report["is_synthetic"]
    assert report["score_spearman"] > 0.999
    assert report["paper_humorous_reproduction"] == "OUT_OF_SCOPE"


def test_all_traits_readouts_and_future_user_comparisons(tmp_path):
    from attentionseekers.io import load_manifest
    from attentionseekers.pipeline import run_pipeline
    model, config = tiny_model()
    data = WS2_ROOT / "samples/ws2_sample.jsonl"
    rows = read_jsonl(data)
    output = run_pipeline(model, CharacterTokenizer(), config, rows, tmp_path / "pipeline",
                          data_path=data, permutations=31, allow_sample=True, require_user=True)
    report = json.loads((output / "summary.json").read_text())
    assert report["is_synthetic"] and report["is_sample"]
    assert len(report["localizations"]) == 12 and len(report["comparisons"]) == 6
    for name in (*report["localizations"], *report["comparisons"]):
        assert load_manifest(output / name)["status"] == "complete"
    index = read_jsonl(output / "activations/index.jsonl")
    assert all(r["response_tokens_used"] == min(150, r["response_tokens_total"]) for r in index)
    metrics = json.loads((output / "E-assigned-resp/metrics.json").read_text())
    assert len(metrics["controls"]["random"]) == 5
    assert report["upstream_reproduction"] == "OUT_OF_SCOPE"
    child = output / "E-assigned-resp/manifest.json"
    child.write_text(child.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(output)


def test_replay_adapter_serialization_and_mismatched_upstream_vectors(tmp_path):
    torch = pytest.importorskip("torch")
    from attentionseekers.io import finish_run, write_json, write_jsonl
    from attentionseekers.repro import replay_scores
    acts = tmp_path / "acts"
    (acts / "weights").mkdir(parents=True)
    reference = tmp_path / "upstream"
    reference.mkdir()
    delta = np.array([[1., 0.], [2., 0.], [3., 0.], [4., 0.]], dtype=np.float32)[None]
    rows = np.stack([delta, -delta])
    np.save(acts / "resp.npy", rows.astype(np.float16))
    np.save(acts / "output_resp.npy", rows.reshape(2, 1, 8))
    np.save(acts / "weights/layer_00.npy", np.eye(8, dtype=np.float32))
    write_json(acts / "config.json", {"model": {"model_name": "tiny-adapter-test", "smh_layer": 0}})
    write_jsonl(acts / "index.jsonl", [{"pole": p, "trait": "humorous"} for p in ("+", "-")])
    finish_run(acts, {"kind": "activations", "mode": "upstream_replay", "is_synthetic": True})
    torch.save(torch.from_numpy(2 * delta.reshape(1, 8)), reference / "humorous_response_avg_diff_attn_pre_o_proj.pt")
    torch.save(torch.from_numpy(2 * delta.reshape(1, 8)), reference / "humorous_response_avg_diff_attn_output.pt")
    report = replay_scores(acts, "humorous", reference, tmp_path)
    assert report["extraction_agreement_pass"] and report["selected_pairs"] == 1
    assert not report["paper_gate_applicable"] and not report["paper_heads_recovered"]
    torch.save(torch.from_numpy((2 * delta[:, ::-1]).copy().reshape(1, 8)), reference / "humorous_response_avg_diff_attn_pre_o_proj.pt")
    report = replay_scores(acts, "humorous", reference, tmp_path)
    assert not report["extraction_agreement_pass"]


def test_missing_upstream_vectors_rejected_before_model_loading(tmp_path, monkeypatch):
    from argparse import Namespace
    from attentionseekers.cli import cmd_reproduce
    import attentionseekers.extract as extraction
    header = "humorous,coherence,prompt,answer\n"
    pos, neg = tmp_path / "pos.csv", tmp_path / "neg.csv"
    pos.write_text(header + "80,90,Prompt+,Answer+\n")
    neg.write_text(header + "10,90,Prompt-,Answer-\n")
    def forbidden(config):
        pytest.fail("Model loading must not start when upstream vector inputs are absent")
    monkeypatch.setattr(extraction, "load_model", forbidden)
    with pytest.raises(FileNotFoundError, match="Required upstream vector absent"):
        cmd_reproduce(Namespace(pos_csv=pos, neg_csv=neg, trait="humorous", threshold=50,
                                config=WS2_ROOT / "configs/qwen.json", upstream_vectors=tmp_path))
