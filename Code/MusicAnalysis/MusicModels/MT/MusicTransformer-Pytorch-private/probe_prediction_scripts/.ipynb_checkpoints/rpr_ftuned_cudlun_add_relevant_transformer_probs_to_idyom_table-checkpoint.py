# merge_transformer_into_cudlun_idyom.py
import pandas as pd
import re

# Inputs/outputs 
idyom_csv = 'probe_prediction_results/cudlun_idyom_probe_probs_midi54_78_with_ics.csv'
transformer_csv = 'probe_prediction_results/rpr_ftuned_cudlun_transformer_probe_probs.csv'
out_csv = 'probe_prediction_results/rpr_ftuned_cudlun_idyom_probs_ics_transformer_probs.csv'

midi_lo, midi_hi = 54, 78

# Load 
idyom = pd.read_csv(idyom_csv)
mt = pd.read_csv(transformer_csv)

# Build join keys (case-insensitive, strip .mid and optional _with_probe) 
def norm_name(s):
    if pd.isna(s): return s
    s = str(s).strip()
    s = re.sub(r'\.mid$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'_with_probe$', '', s, flags=re.IGNORECASE)
    return s.lower()

if 'melody.name' in idyom.columns:
    idyom['join_key'] = idyom['melody.name'].map(norm_name)
elif 'filename' in idyom.columns:
    idyom['join_key'] = idyom['filename'].map(norm_name)
else:
    raise ValueError("IDyOM CSV must contain 'melody.name' or 'filename' to join on.")

if 'filename' in mt.columns:
    mt['join_key'] = mt['filename'].map(norm_name)
else:
    raise ValueError("Transformer CSV must contain a 'filename' column (e.g., 'cudlun95-1.mid').")

# Pick transformer columns for MIDI 54..78 and rename 
wanted = [f'prob_{m}' for m in range(midi_lo, midi_hi + 1)]
available = [c for c in wanted if c in mt.columns]
if not available:
    # fallback: capture any column with an integer and keep those in 54..78
    for c in mt.columns:
        m = re.search(r'(\d+)$', c)
        if m and midi_lo <= int(m.group(1)) <= midi_hi:
            available.append(c)

rename_map = {}
for c in available:
    m = re.search(r'(\d+)$', c)
    midi = int(m.group(1))
    rename_map[c] = f'transformer prob {midi}'

mt_slim = mt[['join_key'] + available].rename(columns=rename_map)

# Ensure numeric
for c in mt_slim.columns:
    if c != 'join_key':
        mt_slim[c] = pd.to_numeric(mt_slim[c], errors='coerce')

# Merge (left: keep all IDyOM rows) 
merged = idyom.merge(mt_slim, on='join_key', how='left')

# IDyOM probs -> IDyOM ICs -> Transformer probs
base = [c for c in merged.columns if c in ('melody.id','melody.name','filename','join_key')]
idyom_probs = [c for c in merged.columns if c.startswith('idyom prob ')]
idyom_ics   = [c for c in merged.columns if c.startswith('idyom ic ')]
mt_probs    = [c for c in merged.columns if c.startswith('transformer prob ')]
other = [c for c in merged.columns if c not in base + idyom_probs + idyom_ics + mt_probs]

ordered = base + idyom_probs + idyom_ics + mt_probs + other
merged = merged[ordered]

merged.to_csv(out_csv, index=False)
print(f"Done! Wrote: {out_csv}")
