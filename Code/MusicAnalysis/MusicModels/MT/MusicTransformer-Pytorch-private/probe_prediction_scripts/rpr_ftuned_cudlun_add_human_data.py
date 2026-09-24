# merge_cudlun_ratings_into_table.py
# Appends trained/untrained ratings (MIDI 54..78) to your merged IDyOM/Transformer CSV.

import re
import pandas as pd
import numpy as np
from pathlib import Path


idyom_mt_csv = "probe_prediction_results/rpr_ftuned_cudlun_merged_idyom_probs_ics_transformer_probs_ics.csv"
cudlun_data  = "cudlun95.data"
out_csv      = "probe_prediction_results/rpr_ftuned_cudlun_merged_idyom_transformer_human.csv"
midi_lo, midi_hi = 54, 78  # inclusive

# load merged IDyOM/Transformer table
df = pd.read_csv(idyom_mt_csv)

# keep a clean, deterministic order (matches your convention)
if "melody.id" in df.columns:
    df = df.sort_values("melody.id").reset_index(drop=True)
else:
    raise ValueError("Expected a 'melody.id' column in the merged CSV.")

# parse cudlun95.data into 8 blocks of 25+25 ratings
with open(cudlun_data, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

lines = raw.splitlines()
stimuli = []  # each: {'name': str, 'trained': [25], 'untrained': [25]}
current = None
num_pat = re.compile(r"[-+]?\d+(?:\.\d+)?")

for ln in lines:
    s = ln.strip()
    if not s:
        continue
    if s.lower().startswith("trained"):  # header line "trained\tuntrained"
        continue
    if s.startswith("#"):                # new stimulus marker, e.g. "#M2 ascending"
        if current is not None and (current["trained"] or current["untrained"]):
            stimuli.append(current)
        current = {"name": s[1:].strip(), "trained": [], "untrained": []}
        continue

    nums = num_pat.findall(s)
    if len(nums) >= 2:
        tr = float(nums[0]); un = float(nums[1])
        current["trained"].append(tr)
        current["untrained"].append(un)

# push last block
if current is not None and (current["trained"] or current["untrained"]):
    stimuli.append(current)

# sanity checks
if len(stimuli) != len(df):
    raise ValueError(f"Row count mismatch: merged CSV has {len(df)} rows but cudlun95.data has {len(stimuli)} stimuli blocks.")

for i, st in enumerate(stimuli, start=1):
    if len(st["trained"]) != (midi_hi - midi_lo + 1) or len(st["untrained"]) != (midi_hi - midi_lo + 1):
        raise ValueError(
            f"Stimulus {i} ('{st['name']}') does not have 25 ratings per group "
            f"(got trained={len(st['trained'])}, untrained={len(st['untrained'])})."
        )

# build rating columns and merge 
trained_cols   = [f"trained rating {m}"   for m in range(midi_lo, midi_hi + 1)]
untrained_cols = [f"untrained rating {m}" for m in range(midi_lo, midi_hi + 1)]

trained_vals   = pd.DataFrame([st["trained"]   for st in stimuli], columns=trained_cols)
untrained_vals = pd.DataFrame([st["untrained"] for st in stimuli], columns=untrained_cols)
avg_cols = [f"avg rating {m}" for m in range(midi_lo, midi_hi + 1)]
avg_vals = pd.DataFrame(
    [
        [(tr + un) / 2 for tr, un in zip(st["trained"], st["untrained"])]
        for st in stimuli
    ],
    columns=avg_cols
)

out = pd.concat([df, trained_vals, untrained_vals, avg_vals], axis=1)
# out = pd.concat([df, trained_vals, untrained_vals], axis=1)

# ---- save ----
out.to_csv(out_csv, index=False)
print(f"Done! Wrote {len(out)} rows with ratings to: {out_csv}")

