"""
Extract IDyOM's predicted probability for the actual note heard at each timestep.
Uses the Music Transformer's detected pitches (from actual_note_probs.csv) as the
ground truth for which pitch column to look up in the IDyOM .dat file.
"""

import os
import pandas as pd
import numpy as np

# ── paths (edit these as needed) ───────────────────────────────────────────────
dat_prefix = '1004'  # change this to match a different .dat file
dat_dir = 'probe_prediction_results'
mt_actual_path = 'probe_prediction_results/rpr_ftuned_manz_transformer_actual_note_probs.csv'

# ── load Transformer actual-note data ──────────────────────────────────────────
df_mt = pd.read_csv(mt_actual_path)
print(f"Loaded Transformer data: {len(df_mt)} rows")

# ── load IDyOM .dat file (find by prefix, use \\?\ prefix on Windows for % in filenames)
dat_candidates = [f for f in os.listdir(dat_dir) if f.startswith(dat_prefix) and f.endswith('.dat')]
if not dat_candidates:
    raise FileNotFoundError(f"No .dat file starting with '{dat_prefix}' found in {dat_dir}")
dat_filename = dat_candidates[0]
dat_fullpath = os.path.join(os.path.abspath(dat_dir), dat_filename)
# Windows needs \\?\ prefix to handle % in filenames
if os.name == 'nt':
    dat_fullpath = '\\\\?\\' + dat_fullpath
print(f"Found IDyOM file: {dat_filename}")

with open(dat_fullpath, 'r') as fh:
    raw_lines = fh.readlines()
header = raw_lines[0].strip().split()
data_rows = [line.strip().split() for line in raw_lines[1:] if line.strip()]
dat_df = pd.DataFrame(data_rows, columns=header)

# Find pitch probability columns (cpitch.NN where NN is an integer)
pitch_cols = [c for c in dat_df.columns if c.startswith('cpitch.') and c.split('.')[-1].isdigit()]
pitch_range = sorted([int(c.split('.')[-1]) for c in pitch_cols])
print(f"Loaded IDyOM data: {len(dat_df)} rows, pitch columns cover {pitch_range[0]}-{pitch_range[-1]}")

# ── verify row counts match ────────────────────────────────────────────────────
if len(df_mt) != len(dat_df):
    print(f"WARNING: Row count mismatch! Transformer has {len(df_mt)}, IDyOM has {len(dat_df)}")
    print("Results may not align correctly. Fix the input data first.")

# ── extract IDyOM probability for each note's actual pitch ─────────────────────
results = []
for i in range(len(df_mt)):
    pitch = int(df_mt.loc[i, 'actual_pitch'])
    col_name = f'cpitch.{pitch}'

    if col_name in dat_df.columns:
        idyom_prob = float(dat_df.loc[i, col_name])
    else:
        print(f"  Warning: pitch {pitch} at row {i} is outside IDyOM range ({pitch_range[0]}-{pitch_range[-1]}), setting to NaN")
        idyom_prob = np.nan

    results.append({
        'filename': df_mt.loc[i, 'filename'],
        'note_index': int(df_mt.loc[i, 'note_index']),
        'actual_pitch': pitch,
        'mt_probability': float(df_mt.loc[i, 'probability']),
        'idyom_probability': idyom_prob
    })

df_out = pd.DataFrame(results)

# ── save ───────────────────────────────────────────────────────────────────────
out_path = 'probe_prediction_results/manz_idyom_actual_note_probs.csv'
df_out.to_csv(out_path, index=False)
print(f"\nSaved {len(df_out)} rows to: {out_path}")
print(df_out.head(10).to_string())
