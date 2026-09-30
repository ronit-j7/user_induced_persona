"""Check extraction against the unmodified reference functions on identical inputs.

This validates implementation agreement on project inputs. The original humorous
experiment and judged-data replay are outside the current WS2 scope.
"""
import ast
from contextlib import nullcontext
import gc
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .extract import capture_encoded
from .heads import head_contributions, spearman
from .io import sha256
from .repro import encode_upstream


def reference_functions():
    import torch
    from tqdm import tqdm
    root = Path(__file__).resolve().parents[1] / "reference_repos/style-modulation-head/src"
    selected = {
        "generate_vec/common.py": {"get_max_layer", "locate_layer_list", "get_attention_config"},
        "generate_vec/generate_vec_head.py": {"get_attn_pre_o_proj_vectors"},
        "generate_vec/generate_vec_block.py": {"get_hidden_block_inputs"},
        "head_analysis/head_contribution/compute.py": {"inner_product_similarity", "compute_head_contributions"},
    }
    namespace = {"torch": torch, "np": np, "gc": gc, "tqdm": tqdm}
    sources = {}
    # Load the exact function definitions, excluding unrelated module startup
    # that demands API credentials even though extraction makes no API calls.
    for relative, names in selected.items():
        source = root / relative
        functions = [n for n in ast.parse(source.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in functions} != names:
            raise ValueError(f"Reference function definitions changed: {relative}")
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        sources[relative] = sha256(source)
    return namespace, sources


def verify_reference_extraction(model, tokenizer, config, rows):
    import torch
    if len(rows) != 2:
        raise ValueError("Supply one positive and one negative matched row")
    reference, sources = reference_functions()
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True) for row in rows]
    answers = [row["forced_response"] for row in rows]
    # The original functions unconditionally synchronize CUDA. CPU tests omit
    # that synchronization only; the extraction/hook arithmetic is untouched.
    context = nullcontext() if model.device.type == "cuda" else patch.object(torch.cuda, "synchronize")
    with context:
        upstream_pre = reference["get_attn_pre_o_proj_vectors"](model, tokenizer, prompts, answers)
        upstream_output = reference["get_hidden_block_inputs"](model, tokenizer, prompts, answers)
    theirs_pre = np.stack([upstream_pre["response_avg"][l].numpy() for l in range(config.num_layers)], axis=1)
    theirs_output = np.stack([upstream_output["attn_output"][l]["response_avg"].numpy()
                             for l in range(config.num_layers)], axis=1)
    own = [capture_encoded(model, config, encode_upstream(tokenizer, {
        "id": rows[i]["id"], "prompt": prompts[i], "answer": answers[i]}, config.max_length),
        readouts=("resp",), compatibility=True) for i in range(2)]
    ours_pre = np.stack([r[0]["resp"].reshape(config.num_layers, -1) for r in own])
    ours_output = np.stack([r[1]["resp"] for r in own])
    np.testing.assert_allclose(ours_pre, theirs_pre, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ours_output, theirs_output, rtol=1e-4, atol=1e-6)
    ours_delta, theirs_delta = ours_pre[0] - ours_pre[1], theirs_pre[0] - theirs_pre[1]
    ours_target, theirs_target = ours_output[0] - ours_output[1], theirs_output[0] - theirs_output[1]
    own_scores, upstream_scores = [], []
    for layer, module in enumerate(model.model.layers):
        weight = module.self_attn.o_proj.weight.detach().float().cpu()
        scores, _ = head_contributions(ours_delta[layer].reshape(config.num_heads, config.head_dim),
                                      weight.numpy(), ours_target[layer])
        actual = reference["compute_head_contributions"](
            torch.from_numpy(theirs_delta[layer]), torch.from_numpy(theirs_target[layer]),
            weight, config.num_heads, config.head_dim)
        np.testing.assert_allclose(scores, actual, rtol=5e-4, atol=1e-5)
        own_scores.append(scores)
        upstream_scores.append(actual)
    own_scores, upstream_scores = np.array(own_scores), np.array(upstream_scores)
    return {"check": "independent_upstream_implementation_agreement", "passed": True,
            "is_synthetic": config.is_synthetic, "rows": [r["id"] for r in rows],
            "model_name": config.model_name, "revision": config.revision,
            "response_window": "full_response_upstream_convention",
            "max_pre_error": float(np.max(np.abs(ours_pre - theirs_pre))),
            "max_output_error": float(np.max(np.abs(ours_output - theirs_output))),
            "max_score_error": float(np.max(np.abs(own_scores - upstream_scores))),
            "score_spearman": spearman(own_scores, upstream_scores),
            "reference_source_sha256": sources, "paper_humorous_reproduction": "OUT_OF_SCOPE"}
