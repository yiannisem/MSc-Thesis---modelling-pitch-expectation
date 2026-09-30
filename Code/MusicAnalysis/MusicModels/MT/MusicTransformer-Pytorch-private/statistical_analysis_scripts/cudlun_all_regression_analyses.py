# rpr_ftuned_cudlun_multi_analysis.py
# One-stop statistical analysis for:
#  - IDyOM IC vs avg rating
#  - Transformer IC vs avg rating
#  - Transformer IC vs IDyOM IC
#  - Multiple regression: avg rating ~ IDyOM IC + Transformer IC

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


CSV_PATH = 'probe_prediction_results/rpr_ftuned_cudlun_merged_idyom_transformer_human.csv'
MIDI_LO, MIDI_HI = 54, 78  # inclusive
OUT_DIR = Path('statistical_analysis_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_columns(df, prefix):
    """Return sorted columns matching e.g. 'idyom ic 54..78' in numeric order."""
    pat = re.compile(rf'^{re.escape(prefix)}\s+(\d+)$')
    pairs = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            note = int(m.group(1))
            if MIDI_LO <= note <= MIDI_HI:
                pairs.append((note, c))
    pairs.sort(key=lambda x: x[0])
    return [c for _, c in pairs]

def make_long(df, idyom_ic_cols, transf_ic_cols, avg_cols):
    """Stack wide probe columns to long format."""
    n_notes = MIDI_HI - MIDI_LO + 1
    long_df = pd.DataFrame({
        'melody.id': df['melody.id'].repeat(n_notes).values,
        'probe_midi': np.tile(np.arange(MIDI_LO, MIDI_HI + 1), len(df)),
        'idyom_ic': df[idyom_ic_cols].to_numpy().reshape(-1),
        'transformer_ic': df[transf_ic_cols].to_numpy().reshape(-1),
        'avg_rating': df[avg_cols].to_numpy().reshape(-1),
    })
    return long_df

def ols_and_corr(x, y, x_name, y_name, tag):
    """Run OLS (y ~ x) + Pearson r, save summary and a scatter+fit plot."""
    # Drop NAs
    valid = pd.notna(x) & pd.notna(y)
    x = pd.Series(x[valid], name=x_name)
    y = pd.Series(y[valid], name=y_name)

    # OLS
    X_const = sm.add_constant(x)
    model = sm.OLS(y, X_const).fit()

    # Pearson correlation
    r, p = pearsonr(x, y)

    # Save text summary
    txt_path = OUT_DIR / f"cudlun_{tag}_summary.txt"
    with open(txt_path, 'w') as f:
        f.write(model.summary().as_text())
        f.write('\n')
        f.write(f"Pearson r = {r:.4f}, p = {p:.3g}\n")
    print(f"[saved] {txt_path}")

    # Scatter + regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.45, s=16)
    # Regression line from model
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model.predict(sm.add_constant(x_line))
    plt.plot(x_line, y_line, linewidth=2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{y_name} vs {x_name}')
    plt.grid(True)
    plt.tight_layout()
    png_path = OUT_DIR / f"cudlun_{tag}_scatter_fit.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[saved] {png_path}")

def multiple_regression(avg, idyom_ic, transf_ic, tag):
    """Run OLS: avg_rating ~ idyom_ic + transformer_ic (with const)."""
    df = pd.DataFrame({'avg_rating': avg, 'idyom_ic': idyom_ic, 'transformer_ic': transf_ic}).dropna()
    X = sm.add_constant(df[['idyom_ic', 'transformer_ic']])
    y = df['avg_rating']
    model = sm.OLS(y, X).fit()

    # Also provide standardized betas for interpretability
    z = df.copy()
    z[['avg_rating', 'idyom_ic', 'transformer_ic']] = (z[['avg_rating', 'idyom_ic', 'transformer_ic']] - 
                                                       z[['avg_rating', 'idyom_ic', 'transformer_ic']].mean()) / \
                                                      z[['avg_rating', 'idyom_ic', 'transformer_ic']].std(ddof=0)
    Xz = sm.add_constant(z[['idyom_ic', 'transformer_ic']])
    yz = z['avg_rating']
    model_z = sm.OLS(yz, Xz).fit()

    txt_path = OUT_DIR / f"cudlun_{tag}_multiple_regression.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Raw-scale regression ===\n")
        f.write(model.summary().as_text())
        f.write("\n\n=== Standardized (beta) regression ===\n")
        f.write(model_z.summary().as_text())
    print(f"[saved] {txt_path}")

def main():
    df = pd.read_csv(CSV_PATH)

    idyom_ic_cols = find_columns(df, "idyom ic")
    transf_ic_cols = find_columns(df, "transformer ic")
    avg_cols       = find_columns(df, "avg rating")

    # Sanity checks
    for name, cols in [('IDyOM IC', idyom_ic_cols), ('Transformer IC', transf_ic_cols), ('Avg Rating', avg_cols)]:
        if len(cols) != (MIDI_HI - MIDI_LO + 1):
            raise ValueError(f"{name}: expected {MIDI_HI - MIDI_LO + 1} columns for MIDI {MIDI_LO}..{MIDI_HI}, got {len(cols)}. "
                             f"Check your column prefixes match the script expectations.")

    long_df = make_long(df, idyom_ic_cols, transf_ic_cols, avg_cols)

    # --- Pairwise analyses ---
    ols_and_corr(
        x=long_df['idyom_ic'],
        y=long_df['avg_rating'],
        x_name='IDyOM Information Content',
        y_name='Average Rating',
        tag='avg_rating_vs_idyom_ic'
    )

    ols_and_corr(
        x=long_df['transformer_ic'],
        y=long_df['avg_rating'],
        x_name='Transformer Information Content',
        y_name='Average Rating',
        tag='avg_rating_vs_transformer_ic'
    )

    ols_and_corr(
        x=long_df['idyom_ic'],
        y=long_df['transformer_ic'],
        x_name='IDyOM Information Content',
        y_name='Transformer Information Content',
        tag='transformer_ic_vs_idyom_ic'
    )

    # --- Multiple regression: avg ~ idyom_ic + transformer_ic ---
    multiple_regression(
        avg=long_df['avg_rating'],
        idyom_ic=long_df['idyom_ic'],
        transf_ic=long_df['transformer_ic'],
        tag='avg_rating_on_both_ics'
    )

    print("\nAll analyses complete.")

if __name__ == "__main__":
    main()
