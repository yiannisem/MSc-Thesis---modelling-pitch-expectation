import pandas as pd
import numpy as np

# Load your merged table
df = pd.read_csv('probe_prediction_results/rpr_ftuned_cudlun_idyom_probs_ics_transformer_probs.csv')

# Find all transformer probability columns (in order)
tf_prob_cols = [col for col in df.columns if col.startswith('transformer prob')]

# For each, compute IC and add as a new column to the right
for idx, prob_col in enumerate(tf_prob_cols):
    ic_col = f'transformer ic {idx+54}'
    df[ic_col] = -np.log2(df[prob_col].replace(0, np.nan))

# Save to a new file
df.to_csv('probe_prediction_results/rpr_ftuned_cudlun_merged_idyom_probs_ics_transformer_probs_ics.csv', index=False)
print("Done! Table saved as rpr_ftuned_cudlun_merged_idyom_probs_ics_transformer_probs_ics.csv")
