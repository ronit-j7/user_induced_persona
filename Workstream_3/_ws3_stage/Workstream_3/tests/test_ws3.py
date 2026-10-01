import numpy as np

from ws3.diagnostic import same_pole_differences
from ws3.factorial import design_matrix, fit_centered, cosine, design_row
from ws3.generate import zero_head_hook
from ws3.jobs import build_jobs, public_record
from ws3.judge import score_logprobs
from ws3.mirroring import mirroring_report
from ws3.rubrics import render


def _scenarios():
    return [
        {"scenario": "code_01", "domain": "coding", "intent": "Reverse a linked list.", "user": "How do I reverse a list?"},
        {"scenario": "emo_01", "domain": "emotional_advice", "intent": "Respond to a friend.", "user": "My friend forgot my birthday."},
    ]


def _variants():
    rows = []
    for scenario, domain in (("code_01", "coding"), ("emo_01", "emotional_advice")):
        for trait in ("E", "A"):
            for pole in ("+", "-"):
                for k in range(3):
                    rows.append({
                        "id": f"{scenario}|U|{trait}|{pole}|k{k}",
                        "set": "user_variant",
                        "scenario": scenario,
                        "domain": domain,
                        "trait": trait,
                        "user_pole": pole,
                        "system": "You are a helpful assistant.",
                        "user": f"styled {trait} {pole} {k} {scenario}",
                        "user_paraphrase": k,
                    })
    return rows


def test_job_count_and_hidden_user_text():
    jobs = build_jobs(_variants(), _scenarios(), samples_per_variant=2, neutral_samples=3)
    assert len(jobs) == 24 * 2 + 2 * 3
    assert len({job["id"] for job in jobs}) == len(jobs)
    assert all(job["system"] == "You are a helpful assistant." for job in jobs)
    record = public_record(jobs[0], "hello", 4)
    assert "user" not in record
    assert record["n_tokens"] == 4


def test_trait_prompt_rejects_user_text():
    prompt = render("E", intent="Reverse a linked list.", answer="Use three pointers.", forbidden=["styled E"])
    assert "Reverse a linked list." in prompt
    assert "Use three pointers." in prompt
    try:
        render("A", intent="Reverse a linked list.", answer="styled E leaked", forbidden=["styled E"])
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("user text was not rejected")


def test_score_logprobs_threshold():
    assert score_logprobs({"80": 0.5, "90": 0.5}) == 85
    assert score_logprobs({" 70": 0.2, "no": 0.8}) is None
    assert score_logprobs({"100": 0.3, "-1": 0.7}) == 100


def test_codes_are_orthogonal_and_cosine_recovers_shared_direction():
    rows = []
    for scenario in ("s1", "s2"):
        for sys_pole in ("+", "-", "0"):
            for user_pole in ("+", "-"):
                rows.append({"scenario": scenario, "sys_pole": sys_pole, "user_pole": user_pole})
    x = design_matrix(rows)
    gram = x.T @ x
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) < 1e-8
    direction = np.zeros(8)
    direction[0] = 1
    z = (x[:, 0] + x[:, 2])[:, None] * direction
    beta, r2 = fit_centered(x, z, np.array([row["scenario"] for row in rows]))
    assert cosine(beta[0], beta[2]) > 0.99
    assert r2["U"] > 0.2 and r2["A"] > 0.2
    assert abs((design_row("+", "+") * design_row("-", "-")).sum()) >= 0


def test_mirroring_mean_and_blocked_status():
    rows = []
    for scenario in ("a", "b"):
        for pole, score in (("+", 80), ("-", 20)):
            rows.append({"scenario": scenario, "domain": "coding", "trait": "E", "pole": pole,
                         "k": 0, "sample": 0, "E": score, "A": 50, "n_tokens": 10, "refused": False})
    report = mirroring_report(rows, resamples=200, seed=0)
    assert report["status"] == "COMPLETE"
    assert abs(report["traits"]["E"]["mirroring"]["mean"] - 60) < 1e-6
    bare = [{**row, "E": None, "A": None} for row in rows]
    assert mirroring_report(bare, resamples=20)["status"] == "BLOCKED_NO_JUDGE_SCORES"


def test_same_pole_pairing():
    rows = []
    acts = []
    for scenario in ("s1", "s2"):
        for k, value in ((0, 1.0), (1, 3.0)):
            rows.append({"set": "sys_twin", "trait": "E", "sys_pole": "+", "sys_paraphrase": k,
                         "scenario": scenario})
            acts.append(np.full((1, 1, 2), value, dtype=np.float32))
    diffs, names = same_pole_differences(np.stack(acts), rows, "E", "+")
    assert names == ["s1", "s2"]
    assert diffs.shape == (2, 1, 1, 2)
    assert np.allclose(diffs, -2)


def test_zero_hook_clears_head_slice():
    import torch
    from torch import nn

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Identity()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    model = Model()
    seen = {}

    def grab(_module, inputs, _output):
        seen["x"] = inputs[0].detach().clone()

    model.model.layers[0].self_attn.o_proj.register_forward_hook(grab)
    hidden = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    with zero_head_hook(model, 0, [1], head_dim=4):
        model.model.layers[0].self_attn.o_proj(hidden)
    assert torch.equal(seen["x"][0, 0, :4], hidden[0, 0, :4])
    assert torch.equal(seen["x"][0, 0, 4:], torch.zeros(4))
