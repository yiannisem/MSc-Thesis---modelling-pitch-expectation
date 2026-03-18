# takes new_ftuned_rpr_schell_transformer_probe_probs_norm.csv and adds the required diatonic pitches to the already existing idyom table

import pandas as pd

# Load files
idyom_df = pd.read_csv('probe_prediction_results/schellenberg_optimal_viewpoint_idyom_selected_probe_probs_with_ics.csv')
transformer_df = pd.read_csv('probe_prediction_results/rpr_ftuned_schell_transformer_probe_probs_norm.csv')
xlsx_df = pd.read_excel('schellenberg_midi_notes_predicted.xlsx')

# Build the new columns here
transformer_cols = []

for frag_num, frag_col in enumerate(xlsx_df.columns, start=1):
    # Match to corresponding row in transformer CSV
    midi_filename = f'schell96-{frag_num}.mid'
    tf_row = transformer_df[transformer_df['filename'] == midi_filename]
    midi_pitches = xlsx_df[frag_col].dropna().astype(int).tolist()
    tf_probs = []
    for midi in midi_pitches:
        col = f'norm_prob_{midi}'
        # for missing columns
        val = tf_row[col].values[0] if col in tf_row.columns else float('nan')
        tf_probs.append(val)
    transformer_cols.append(tf_probs)

# add transformer_cols as new columns to idyom_df
max_n = max(len(probs) for probs in transformer_cols)
for i in range(max_n):
    colname = f'transformer probability {i+1}'
    # Pad with NaN if some fragments have fewer probes
    idyom_df[colname] = [row[i] if i < len(row) else float('nan') for row in transformer_cols]

# Save
idyom_df.to_csv('probe_prediction_results/schell_transformer_optimal_viewpoint_idyom_probs_ics_transformer_probs.csv', index=False)
print("Done! Table saved as schell_transformer_optimal_viewpoint_idyom_probs_ics_transformer_probs.csv")
