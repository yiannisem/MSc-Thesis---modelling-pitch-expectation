# Merge human expectation ratings from manzetal92.data into the
# combined IDyOM/Transformer actual-note probability table.

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
model_csv  = 'probe_prediction_results/manz_idyom_actual_note_probs.csv'
human_data = 'manzetal92.data'
out_csv    = 'probe_prediction_results/manz_merged_idyom_transformer_human.csv'

# ── load model probabilities ───────────────────────────────────────────────────
df = pd.read_csv(model_csv)
print(f"Loaded model data: {len(df)} rows")

# ── load human ratings (skip comment lines starting with #) ────────────────────
human_rows = []
with open(human_data, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('chorale'):
            continue
        parts = line.split()
        human_rows.append({
            'chorale': int(parts[0]),
            'note': int(parts[1]),
            'human_weighted': float(parts[3]),
        })

df_human = pd.DataFrame(human_rows)
print(f"Loaded human data: {len(df_human)} rows")

# ── verify row counts match ────────────────────────────────────────────────────
if len(df) != len(df_human):
    raise ValueError(f"Row count mismatch: model has {len(df)}, human has {len(df_human)}")

# ── merge (row-by-row, same order) ─────────────────────────────────────────────
out = pd.concat([df, df_human.reset_index(drop=True)], axis=1)

# ── save ───────────────────────────────────────────────────────────────────────
out.to_csv(out_csv, index=False)
print(f"\nSaved {len(out)} rows to: {out_csv}")
print(out.head(10).to_string())
