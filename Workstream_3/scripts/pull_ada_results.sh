#!/usr/bin/env bash
# Copy Ada job outputs into results/ada/ in this repo. That directory is gitignored.
# Skips the o_proj weight dump under acts/weights (a copy of the model, not a result).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/results/ada"
KEY="${ADA_KEY:-$HOME/.ssh/ada_uip}"
mkdir -p "$DEST/yajat" "$DEST/bhavya"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes)

"${SSH[@]}" yajat.lakhanpal@ada-gw1.iiit.ac.in 'bash -s' << 'EOF' | tar -xzf - -C "$DEST/yajat"
set -euo pipefail
cd "$HOME/ap_store/results"
args=()
[ -d jobA ] && args+=(jobA)
[ -d jobB ] && args+=(jobB)
tar --exclude='jobA/acts/weights' --exclude='jobB/acts/weights' -czf - "${args[@]}"
EOF

"${SSH[@]}" yajat.lakhanpal@ada-gw1.iiit.ac.in 'bash -s' << 'EOF' | tar -xzf - -C "$DEST/yajat"
set -euo pipefail
cd "$HOME/user_induced_persona"
args=()
for f in ap-proj-ap-proj-49467.out ap-proj-ap-proj-49468.out; do
  [ -f "$f" ] && args+=("$f")
done
if ((${#args[@]})); then tar -czf - "${args[@]}"; else tar -czf - --files-from /dev/null; fi
EOF

"${SSH[@]}" bhavya.ahuja@ada-gw1.iiit.ac.in 'bash -s' << 'EOF' | tar -xzf - -C "$DEST/bhavya"
set -euo pipefail
cd "$HOME/ap_store/results"
tar --exclude='jobB/acts/weights' -czf - .
EOF

echo "pulled $(date -Is) -> $DEST"
du -sh "$DEST/yajat" "$DEST/bhavya" 2>/dev/null || true
