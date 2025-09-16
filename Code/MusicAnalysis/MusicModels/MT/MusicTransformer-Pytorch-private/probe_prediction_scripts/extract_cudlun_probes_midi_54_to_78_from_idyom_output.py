# Extract cpitch probabilities for MIDI 54..78 from the probe (last note) of each cudlun stimulus.

import pandas as pd

# Config
dat_path = 'probe_prediction_results/1003-cpitch-cpitch-100_200_300-nil-melody-nil-1-both-nil-t-nil-c-nil-t-t-x-3.dat'
out_csv  = 'probe_prediction_results/cudlun_idyom_probe_probs_midi54_78.csv'
midi_lo, midi_hi = 54, 78  # inclusive

# Load IDyOM output 
df = pd.read_csv(dat_path, sep=r'\s+', comment='#')

# Pick the probe row (last event) per melody
last_rows = (
    df.sort_values('note.id', kind='mergesort')  # stable
      .groupby('melody.id', as_index=False)
      .tail(1)
)

# Build the output table
wanted_cols = [f'cpitch.{m}' for m in range(midi_lo, midi_hi + 1)]
for c in wanted_cols:
    if c not in last_rows.columns:
        last_rows[c] = float('nan')

out_cols = ['melody.id', 'melody.name'] + wanted_cols
out_df = last_rows[out_cols].sort_values(['melody.name', 'melody.id']).reset_index(drop=True)

# Rename the cpitch.* columns for output only
rename_map = {f'cpitch.{m}': f'idyom prob {m}' for m in range(midi_lo, midi_hi + 1)}
out_df = out_df.rename(columns=rename_map)

out_df.to_csv(out_csv, index=False)
print(f"Done! Table saved as {out_csv}")
