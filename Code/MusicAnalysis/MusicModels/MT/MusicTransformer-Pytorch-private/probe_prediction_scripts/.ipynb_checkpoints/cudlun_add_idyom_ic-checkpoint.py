# takes the big table with only the idyom predictions in it and adds the ic

# import pandas as pd
# import numpy as np

# df = pd.read_csv('cudlun_idyom_probe_probs_midi54_78.csv')

# prob_indices = [i for i, col in enumerate(df.columns) if 'idyom probability' in col]
# prob_cols = [df.columns[i] for i in prob_indices]

# # Build DataFrame with fragment + all probs
# cols = ['fragment'] + prob_cols
# out_df = df[cols].copy()

# # Rename probability columns so they're numbered from 1
# new_prob_cols = [f'idyom probability {i+1}' for i in range(len(prob_cols))]
# rename_dict = {old: new for old, new in zip(prob_cols, new_prob_cols)}
# out_df = out_df.rename(columns=rename_dict)

# # Add IC columns
# for idx, prob_col in enumerate(new_prob_cols):
#     ic_col = f'idyom ic {idx+1}'
#     prob_vals = out_df[prob_col]
#     out_df[ic_col] = -np.log2(prob_vals.replace(0, np.nan))

# out_df.to_csv('prediction_results/schellenberg_idyom_selected_probe_probs_with_ics.csv', index=False)
# print("Done! Table saved as schellenberg_idyom_selected_probe_probs_with_ics.csv")

# Add ic columns for MIDI 54..78 to the cudlun probe probs table

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
