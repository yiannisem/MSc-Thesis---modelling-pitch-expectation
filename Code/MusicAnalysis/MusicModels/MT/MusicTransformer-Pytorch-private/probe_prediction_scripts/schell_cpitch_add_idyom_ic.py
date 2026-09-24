# takes the big table with only the idyom predictions in it and adds the ic

import pandas as pd
import numpy as np

df = pd.read_csv('probe_prediction_results/schellenberg_optimal_viewpoint_idyom_selected_probe_probs.csv')

prob_indices = [i for i, col in enumerate(df.columns) if 'idyom probability' in col]
prob_cols = [df.columns[i] for i in prob_indices]

# Build DataFrame with fragment + all probs
cols = ['fragment'] + prob_cols
out_df = df[cols].copy()

# Rename probability columns so they're numbered from 1
new_prob_cols = [f'idyom probability {i+1}' for i in range(len(prob_cols))]
rename_dict = {old: new for old, new in zip(prob_cols, new_prob_cols)}
out_df = out_df.rename(columns=rename_dict)

# Add IC columns
for idx, prob_col in enumerate(new_prob_cols):
    ic_col = f'idyom ic {idx+1}'
    prob_vals = out_df[prob_col]
    out_df[ic_col] = -np.log2(prob_vals.replace(0, np.nan))

out_df.to_csv('probe_prediction_results/schellenberg_optimal_viewpoint_idyom_selected_probe_probs_with_ics.csv', index=False)
print("Done! Table saved as schellenberg_optimal_viewpoint_idyom_selected_probe_probs_with_ics.csv")
