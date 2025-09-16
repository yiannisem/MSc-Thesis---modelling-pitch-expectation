# Get the specified diatonic pitches and put them in a new table

import pandas as pd
import re

dat_path = 'probe_prediction_results/1002-cpitch-optimal-viewpoint-100_200_300-nil-melody-nil-1-both-nil-t-nil-c-nil-t-t-x-3.dat'
xlsx_path = 'schellenberg_midi_notes_predicted.xlsx'

# Read files
dat_df = pd.read_csv(dat_path, sep='\s+', comment='#')
xlsx_df = pd.read_excel(xlsx_path)

# Find the last note row for each melody.id
last_rows = dat_df.sort_values('note.id').groupby('melody.id').tail(1)

# Make a mapping from melody.name to its last row
melody_name_to_row = last_rows.set_index('melody.name')

output_rows = []
fragment_cols = list(xlsx_df.columns)

for frag_col in fragment_cols:
    # Extract fragment number from column name (handles both "#fragment 1" and "#fragment1" etc)
    frag_num = int(re.search(r'(\d+)', frag_col).group(1))
    # Compose expected melody name
    expected_name = f'schell96-{frag_num}_with_probe'
    if expected_name not in melody_name_to_row.index:
        print(f"WARNING: {expected_name} not found in .dat for {frag_col}")
        continue  # or raise error
    melody_row = melody_name_to_row.loc[expected_name]
    midi_pitches = xlsx_df[frag_col].dropna().astype(int).tolist()
    values = []
    for midi in midi_pitches:
        colname = f'cpitch.{midi}'
        values.append(melody_row[colname] if colname in melody_row else float('nan'))
    output_rows.append([frag_num] + values)

# Sort output by fragment number!
output_rows = sorted(output_rows, key=lambda x: x[0])

# Add headers
max_len = max(len(row) - 1 for row in output_rows)
header = ['fragment'] + ['idyom probability'] * max_len

for row in output_rows:
    while len(row) < max_len + 1:
        row.append('')

out_df = pd.DataFrame(output_rows, columns=header)
out_df.to_csv('probe_prediction_results/schellenberg_optimal_viewpoint_idyom_selected_probe_probs.csv', index=False)

print("Done! Table saved as schellenberg_optimal_viewpoint_idyom_selected_probe_probs.csv")
