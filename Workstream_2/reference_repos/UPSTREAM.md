# Imported reference repositories

These are unmodified source snapshots, imported with `git archive HEAD` so
they can be committed directly in this repository. Neither directory contains
a nested `.git` directory or submodule. Keep the upstream licenses in place.

| Directory | Upstream | Commit | License | Why included |
| --- | --- | --- | --- | --- |
| `style-modulation-head` | https://github.com/Omusubi0123/style-modulation-head | `532da0151b319efa99145cdad88035c889d72ef3` | MIT, see its `LICENSE` | Izawa et al. SMH reproduction and score reference. |
| `persona_vectors` | https://github.com/safety-research/persona_vectors | `b8e0f044fe2410a6fad579f38324f03f13b4e917` | Apache 2.0, see its `LICENSE` | Trait data and code that the SMH repository cites as a dependency. |

The Persona Vectors snapshot includes its upstream `dataset.zip` (about 58 MiB)
because the request was to import the full reference repository. Experiment
outputs, downloaded model weights, environments, and API keys are not included.
Our implementation in `attentionseekers/` is separate from these snapshots.

To verify the imported files against fresh upstream clones of these exact
commits, run:

```bash
uv run python scripts/verify_reference_imports.py \
  --style-source /path/to/style-modulation-head \
  --persona-source /path/to/persona_vectors
```
