# takes the full transformer probability predictions, extracts the wanted midi range and normalises

import pandas as pd
import numpy as np

csv_path = "probe_prediction_results/transfinidyom_schell_probe_probs.csv"
df = pd.read_csv(csv_path)

# Extract only the columns for MIDI 47–84
probe_cols = [f'prob_{i}' for i in range(47, 85)]
df_probes = df[probe_cols]

# Normalize those probabilities so each row sums to 1
probs = df_probes.values
probs_norm = probs / probs.sum(axis=1, keepdims=True)

# Make a new DataFrame for normalized probs only
norm_cols = [f'norm_prob_{i}' for i in range(47, 85)]
df_final = pd.DataFrame(
    {'filename': df['filename'],
     **{col: probs_norm[:, idx] for idx, col in enumerate(norm_cols)}}
)

df_final.to_csv("probe_prediction_results/transfinidyom_schell_probe_probs_norm.csv", index=False)
print("Saved normalized probabilities to: transfinidyom_schell_probe_probs_norm.csv")
