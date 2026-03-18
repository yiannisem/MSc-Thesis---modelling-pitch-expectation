import pandas as pd
import numpy as np

in_csv  = 'probe_prediction_results/cudlun_idyom_probe_probs_midi54_78.csv'
out_csv = 'probe_prediction_results/cudlun_idyom_probe_probs_midi54_78_with_ics.csv'
midi_lo, midi_hi = 54, 78

df = pd.read_csv(in_csv)

# The probability columns should already exist with these names
prob_cols = [f'idyom prob {m}' for m in range(midi_lo, midi_hi + 1)]

# Ensure numeric (coerce any weird strings to NaN)
for c in prob_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Add IC columns: ic = -log2(p), with p==0 -> NaN
for m in range(midi_lo, midi_hi + 1):
    prob_col = f'idyom prob {m}'
    ic_col = f'idyom ic {m}'
    df[ic_col] = -np.log2(df[prob_col].replace(0, np.nan))

df.to_csv(out_csv, index=False)
print(f"Done! Saved ICs to {out_csv}")
