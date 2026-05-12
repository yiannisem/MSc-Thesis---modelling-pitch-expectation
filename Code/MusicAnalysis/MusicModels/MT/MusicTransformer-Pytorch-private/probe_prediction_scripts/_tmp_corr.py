import pandas as pd
from scipy.stats import pearsonr

df1 = pd.read_csv('probe_prediction_results/manz_idyom_actual_note_probs.csv')

# Load human data manually to avoid dtype issues
human_rows = []
with open('manzetal92.data', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('chorale'):
            continue
        parts = line.split()
        human_rows.append({
            'avg': float(parts[2]),
            'wght': float(parts[3]),
            'best': float(parts[4]),
        })
df2 = pd.DataFrame(human_rows)

df = pd.concat([df1, df2.reset_index(drop=True)], axis=1)

# Drop missing
valid = df.dropna(subset=['idyom_ic', 'avg', 'wght', 'best']).copy()

print("Correlations against IDyOM Information Content (ALL NOTES):")
print("Average rating:", pearsonr(valid['idyom_ic'], valid['avg'])[0])
print("Weighted rating:", pearsonr(valid['idyom_ic'], valid['wght'])[0])

valid_no1 = valid[valid['note_index'] > 0].copy()
print("\nCorrelations against IDyOM Information Content (NO FIRST NOTE):")
print("Average rating:", pearsonr(valid_no1['idyom_ic'], valid_no1['avg'])[0])
print("Weighted rating:", pearsonr(valid_no1['idyom_ic'], valid_no1['wght'])[0])
