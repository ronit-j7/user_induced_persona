# AttentionSeekers

This repository studies whether assistant personality mirroring from a user's
style uses the same attention heads as a system-assigned persona. The
mid-submission scope is Qwen2.5-7B-Instruct and Extraversion/Agreeableness.

[INTERFACES.md](INTERFACES.md) is the cross-workstream data, API, and artifact
contract. [Workstream 2 plan](Context_Documents/workstream_2_implementation.md)
tracks the implementation and remaining GPU/data gates. The current base data
contain 20 neutral scenarios; the labeled WS2 sample data exercise the code
while Workstream 1 builds the real variants and fixed responses.

Start with `uv sync --extra model --extra test`, then run the validation and
smoke commands in `INTERFACES.md`. All experiment outputs go in ignored
`results/` directories. A successful smoke run checks software integration;
it is not a replication claim.
