import numpy as np
import pytest

from ws3.directions import assign_folds, build_directions, load_directions, random_triple, save_directions
from ws3.generate import read_jsonl
from ws3.intervention_report import build_report
from ws3.project import JOB_ARMS, build_projection_jobs, parse_arm

from test_ws3 import _tiny_model


def _capture(model, layer=0):
    seen = {}
    model.model.layers[layer].self_attn.o_proj.register_forward_hook(
        lambda _m, inputs, _o: seen.__setitem__("x", inputs[0].detach().clone()))
    return seen


def test_projection_hook_identity_and_removal():
    import torch
    from ws3.generate import projection_hook
    model = _tiny_model()
    seen = _capture(model)
    hidden = torch.randn(2, 3, 12)
    u = np.zeros(8, dtype=np.float32)
    u[1] = 1.0
    mu0 = np.full(8, 0.5, dtype=np.float32)
    with projection_hook(model, {0: ([0, 2], u, mu0)}, 0.0, head_dim=4):
        model.model.layers[0].self_attn.o_proj(hidden)
    assert torch.equal(seen["x"], hidden)
    with projection_hook(model, {0: ([0, 2], u, mu0)}, 1.0, head_dim=4):
        model.model.layers[0].self_attn.o_proj(hidden)
    # Column 1 (head 0, dim 1) is pulled to mu0; untouched head 1 and other dims stay.
    assert torch.allclose(seen["x"][..., 1], torch.full((2, 3), 0.5))
    assert torch.equal(seen["x"][..., 4:8], hidden[..., 4:8])
    assert torch.equal(seen["x"][..., 9:], hidden[..., 9:])
    with projection_hook(model, {0: ([0, 2], u, mu0)}, 2.0, head_dim=4):
        model.model.layers[0].self_attn.o_proj(hidden)
    assert torch.allclose(seen["x"][..., 1], 1.0 - hidden[..., 1], atol=1e-6)
    model.model.layers[0].self_attn.o_proj(hidden)
    assert torch.equal(seen["x"], hidden), "hook must be removed on exit"


def test_projection_hook_rejects_bad_direction():
    from ws3.generate import projection_hook
    model = _tiny_model()
    with pytest.raises(ValueError):
        with projection_hook(model, {0: ([0], np.ones(4, dtype=np.float32), np.zeros(4)), }, 1.0, head_dim=4):
            pass


def test_folds_cover_every_domain_evenly():
    rows = [{"scenario": f"{d}_{i}", "domain": d} for d in ("a", "b", "c", "d") for i in range(5)]
    folds = assign_folds(rows, 4)
    sizes = [sum(1 for f in folds.values() if f == k) for k in range(4)]
    assert sizes == [5, 5, 5, 5]
    for k in range(4):
        domains = {name.split("_")[0] for name, f in folds.items() if f == k}
        assert len(domains) == 4


def test_random_triple_excludes_smh():
    for seed in range(20):
        triple = random_triple(28, (2, 4, 27), seed)
        assert len(triple) == 3 and not set(triple) & {2, 4, 27}


def _synthetic_bundle(layers=2, heads=4, dim=3, scenarios=8):
    rng = np.random.default_rng(0)
    planted = rng.standard_normal((layers, heads, dim)).astype(np.float32)
    rows, acts = [], []
    for s in range(scenarios):
        name, domain = f"s{s}", "ab"[s % 2]
        base = rng.standard_normal((layers, heads, dim)).astype(np.float32)
        rows.append({"set": "neutral", "trait": "none", "scenario": name, "domain": domain,
                     "user_paraphrase": 0, "sys_paraphrase": 0, "user_pole": "0", "sys_pole": "0",
                     "system": "S", "user": f"n{s}", "forced_response": f"r{s}"})
        acts.append(base)
        for k in range(2):
            for pole, sign in (("+", 1), ("-", -1)):
                rows.append({"set": "user_variant", "trait": "E", "scenario": name, "domain": domain,
                             "user_paraphrase": k, "sys_paraphrase": 0, "user_pole": pole, "sys_pole": "0",
                             "system": "S", "user": f"u{s}{k}{pole}", "forced_response": f"r{s}"})
                acts.append(base + sign * planted + 0.01 * rng.standard_normal(base.shape).astype(np.float32))
    weights = [np.eye(heads * dim, dtype=np.float32) for _ in range(layers)]
    return np.stack(acts), rows, weights, planted


def test_directions_recover_planted_contrast(tmp_path, monkeypatch):
    import ws3.directions as directions
    monkeypatch.setattr(directions, "SMH_LAYER", 1)
    monkeypatch.setattr(directions, "SMH_HEADS", (0, 2))
    acts, rows, weights, planted = _synthetic_bundle()
    groups, stats = directions.build_directions(acts, rows, weights, n_folds=2, top_k=3)
    expected = np.concatenate([planted[1, 0], planted[1, 2]])
    expected /= np.linalg.norm(expected)
    for fold in (0, 1):
        heads, u, mu0 = groups["smh"][fold][1]
        assert heads == [0, 2]
        assert float(np.dot(u, expected)) > 0.999
        assert mu0.shape == u.shape
    save_directions(groups, stats, tmp_path / "d")
    loaded, folds = load_directions(tmp_path / "d")
    assert set(folds) == {f"s{i}" for i in range(8)}
    assert np.allclose(loaded["smh"][1][1][1], groups["smh"][1][1][1])
    assert sum(len(h) for h in stats["per_fold"]["0"]["top10"].values()) == 3


def _real_jobs(trait="E", **kwargs):
    return build_projection_jobs(read_jsonl("Data_Creation/data/user_variants.jsonl"),
                                 read_jsonl("Data_Creation/data/factorial.jsonl"),
                                 read_jsonl("Data_Creation/data/scenarios.jsonl"), trait=trait, **kwargs)


def test_projection_jobs_from_real_data():
    jobs = _real_jobs()
    mirror = [j for j in jobs if j["kind"] == "user_variant"]
    cells = [j for j in jobs if j["kind"] == "factorial"]
    assert len(mirror) == 120 and len(cells) == 80
    assert all(j["trait"] == "E" for j in jobs)
    assert {(j["sys_pole"], j["pole"]) for j in cells} == {("+", "+"), ("+", "-"), ("-", "+"), ("-", "-")}
    assert all(j["system"] != "You are a helpful assistant." for j in cells)
    a_jobs = _real_jobs("A", include_factorial=False)
    assert len(a_jobs) == 120 and {j["trait"] for j in a_jobs} == {"A"}


def test_public_record_keeps_sys_pole_and_hides_text():
    from ws3.jobs import public_record
    cell = next(j for j in _real_jobs() if j["kind"] == "factorial")
    record = public_record(cell, "reply", 3)
    assert record["sys_pole"] == cell["sys_pole"]
    assert "user" not in record and "system" not in record


def test_arm_names_parse():
    for arms in JOB_ARMS.values():
        for arm in arms:
            parse_arm(arm)
    assert parse_arm("smh_l2") == {"kind": "proj", "group": "smh", "lam": 2.0, "trait": "E"}
    assert parse_arm("smh_zero")["kind"] == "zero"
    assert parse_arm("baselineA") == {"kind": "none", "trait": "A"}
    assert parse_arm("resid19_l1")["group"] == "resid19"
    with pytest.raises(ValueError):
        parse_arm("nonsense")


def test_residual_hook_identity_and_removal():
    import torch
    from ws3.residual import layer_output_hook
    model = _tiny_model()
    layer = model.model.layers[0]
    layer.forward = lambda x: (x * 1.0, None)
    hidden = torch.randn(1, 2, 4)
    u = np.array([0, 0, 1, 0], dtype=np.float32)
    with layer_output_hook(model, 0, u, np.zeros(4, dtype=np.float32), 0.0):
        out = layer(hidden)
    assert torch.equal(out[0], hidden)
    with layer_output_hook(model, 0, u, np.zeros(4, dtype=np.float32), 1.0):
        out = layer(hidden)
    assert torch.all(out[0][..., 2] == 0)
    assert torch.equal(out[0][..., :2], hidden[..., :2])


def test_residual_directions_share_folds(tmp_path):
    from ws3.directions import add_residual_group
    acts, rows, _weights, planted = _synthetic_bundle()
    resid = acts.reshape(len(acts), -1)
    stats = {"folds": assign_folds([r for r in rows if r["set"] == "user_variant"], 2)}
    groups, stats = add_residual_group({}, stats, resid, rows, 19)
    heads, u, mu0 = groups["resid19"][0][19]
    assert heads == [] and u.shape == (resid.shape[1],)
    assert float(np.dot(u, planted.ravel() / np.linalg.norm(planted))) > 0.999


def _cell_rows(scenario, assigned, user, offset=0.0):
    rows = []
    for sys_pole, a in (("+", 1), ("-", -1)):
        for pole, b in (("+", 1), ("-", -1)):
            rows.append({"scenario": scenario, "kind": "factorial", "sys_pole": sys_pole, "pole": pole,
                         "trait": "E", "E": 50 + offset + a * assigned / 2 + b * user / 2,
                         "coherence": 90, "refused": False, "n_tokens": 100})
    return rows


def _scores(m_gap, assigned, user):
    rows = []
    for s in range(6):
        for pole, sign in (("+", 1), ("-", -1)):
            rows.append({"scenario": f"s{s}", "kind": "user_variant", "pole": pole, "trait": "E",
                         "E": 50 + sign * m_gap / 2 + s, "coherence": 90, "refused": False, "n_tokens": 100})
        rows += _cell_rows(f"s{s}", assigned, user, offset=s)
    return rows


def test_intervention_report_reduction_and_factorial():
    report = build_report({"A": {"baseline": _scores(10, 40, 6), "smh_l1": _scores(4, 38, 2)}}, resamples=200)
    arm = report["jobs"]["A"]["smh_l1"]
    assert abs(arm["mirroring"]["mean"] - 4) < 1e-9
    assert abs(arm["mirroring_vs_baseline"]["mean"] + 6) < 1e-9
    assert abs(arm["mirroring_reduction"]["value"] - 0.6) < 1e-9
    assert abs(arm["factorial_assigned"]["mean"] - 38) < 1e-9
    assert abs(arm["factorial_user"]["mean"] - 2) < 1e-9
    assert abs(arm["factorial_conflict"]["mean"] - 36) < 1e-9
    assert abs(arm["factorial_assigned_vs_baseline"]["mean"] + 2) < 1e-9
    assert report["jobs"]["A"]["baseline"]["mirroring_reduction"]["value"] == 0


def test_pairwise_helpers():
    from ws3.pairwise import build_pairs, choice_probability, render_pair, summarize
    assert choice_probability({"A": 0.6, " B": 0.2, "x": 0.2}) == pytest.approx(0.75)
    assert choice_probability({"x": 1.0}) is None
    records = []
    for scenario in ("s1", "s2"):
        for pole in ("+", "-"):
            records.append({"kind": "user_variant", "trait": "A", "scenario": scenario, "k": 0, "sample": 0,
                            "pole": pole, "domain": "coding", "intent": "Do X.", "text": f"{pole} reply"})
    pairs = build_pairs(records, "A")
    assert len(pairs) == 2 and pairs[0]["high"] == "+ reply"
    prompt = render_pair("A", "Do X.", "first", "second")
    assert "first" in prompt and "second" in prompt and "agreeable" in prompt
    results = [{"scenario": "s1", "k": 0, "p_high_order1": 0.9, "p_high_order2": 0.7, "p_high": 0.8},
               {"scenario": "s2", "k": 0, "p_high_order1": 0.4, "p_high_order2": 0.6, "p_high": 0.5}]
    summary = summarize(results, resamples=100)
    assert summary["mean_p_high"] == pytest.approx(0.65)
    assert summary["order_consistency"] == 0.5
