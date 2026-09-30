# Assigned-persona refresh: 2026-10-01

## Scope

The updated team context excludes the original humorous experiment and
judged-data replay. They are **OUT_OF_SCOPE**, not outstanding WS2 tasks.
WS2's reference-function validation already passed on a full-Qwen E twin pair;
the tested model inputs are unchanged and that validation was retained.

## Input audit and reuse

Exact comparison against the September 30 completed activation index found:

- All 10 E system prompts are unchanged; retain the three E localizations.
- All 10 A system prompts changed: each adds "Still help the user fully with
  their request." and the k3 positive/negative pair is reworded.
- All 20 canonical user messages, fixed responses, domains and the neutral
  system are unchanged. Neutral first activations and control groups are retained.
- WS1 user variants are available for WS3. This refresh does not execute user
  localization, behavioral generation or judging.

Only the 200 A system twins were extracted again. The combined 420-row
activation bundle reuses 200 E and 20 neutral rows, verified bit identical for
all three readouts. It records each row's activation origin, old/new manifest
hashes, current prompt provenance and exactly matching projection weights.
Only the three A assigned localizations were computed; saved E reports remain
in the original completed run. All current system texts in the combined index
match the latest WS1 prompt file.

## Results

Official Qwen2.5-7B-Instruct, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bf16 on the RTX 3090, full fixed
responses with a first-150-token response readout. Each localization uses 100
matched pairs, 20 independent scenario clusters and 19,999 sign flips.

| Readout | Rank of head 2 | Rank of head 4 | Rank of head 27 | Heads with BH q < 0.05 / 784 |
| --- | --- | --- | --- | --- |
| first | 2 | 3 | 1 | 688 |
| response | 1 | 3 | 2 | 703 |
| user | 1 | 3 | 2 | 722 |

Ranks refer to zero-indexed layer 19. The `user` readout here averages user-token
activations under **system-assigned** A contrasts; it is not a user-style study.
The neutral highest-norm group stays [0, 2, 17]; all five seeded random groups
and their saved neutral norms are unchanged. Heatmaps, SNR, p/q arrays, deltas
and metrics are complete. **20 tests passed without skips** in the isolated
locked uv environment on the hard disk.

## Handoff

- Latest A localizations and combined activations:
  `results/ws2-refresh/A-system-prompts-2026-10-01-retry/`.
- Unchanged E reports and reference agreement:
  `results/ws1-verification/full-qwen-gpu-retry/`.
- Compact current metrics, absolute artifact paths and immutable manifest
  hashes: `Workstream_2/docs/assigned_refresh_2026-10-01.json`.
- The September 30 A reports are historical and superseded by this refresh.
  Historical completed bundles have not been modified.

All large artifacts stay under
`/media/gaurav/Data21/eshaan/results/user_induced_persona` on the hard disk.
The source and portable compact summaries remain in `Workstream_2/`.

The refresh entry point is `Workstream_2.scripts.refresh_assigned`; its command
and guards are documented in the root `INTERFACES.md`.
