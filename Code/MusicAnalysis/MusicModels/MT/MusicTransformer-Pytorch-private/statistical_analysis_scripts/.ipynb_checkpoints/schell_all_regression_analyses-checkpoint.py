# schell_multi_analysis_ic_savelike_cudlun.py
# Adapted from the CUDLUN multi-analysis script:
# Saves *_summary.txt, *_scatter_fit.png, and a multiple_regression txt file.

import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Point to Schellenberg CSV (with 'idyom ic k', 'transformer ic k', 'human rating k', and optional 'fragment'):
CSV_PATH = 'probe_prediction_results/rpr_ftuned_schell_merged_idyom_transformer_human.csv'

# Match the CUDLUN script's output structure:
OUT_DIR = Path('statistical_analysis_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Column prefixes (case-insensitive)
IDYOM_IC_PREFIX        = 'idyom ic'
TRANSFORMER_IC_PREFIX  = 'transformer ic'
HUMAN_RATING_PREFIX    = 'human rating'



def find_columns(df, prefix, flags=re.I):
    """Return {probe_index:int -> column_name:str} for 'prefix <index>' (case-insensitive)."""
    rx = re.compile(rf'^{re.escape(prefix)}\s+(\d+)$', flags)
    mapping = {}
    for c in df.columns:
        m = rx.match(c.strip())
        if m:
            mapping[int(m.group(1))] = c
    return mapping


def ensure_id_col(df):
    """Use 'fragment' if present; otherwise add a synthetic 'row_id'."""
    if 'fragment' in df.columns:
        return df, 'fragment'
    df = df.copy()
    df['row_id'] = np.arange(len(df)) + 1
    return df, 'row_id'


def to_long(df, fam_y, y_label, fam_x1, x1_label, fam_x2=None, x2_label=None, id_col='fragment'):
    """
    Build long-form DF aligned on overlapping probe indices.
    Uses explicit labels so downstream code can refer to the intended names.
    """
    common = set(fam_y).intersection(fam_x1)
    if fam_x2 is not None:
        common = common.intersection(fam_x2)
    idxs = sorted(common)
    if not idxs:
        raise ValueError("No overlapping probe indices across the requested families.")

    n = len(idxs)
    data = {
        id_col: df[id_col].repeat(n).values,
        'probe_idx': np.tile(idxs, len(df)),
        y_label:  df[[fam_y[i]  for i in idxs]].to_numpy().reshape(-1),
        x1_label: df[[fam_x1[i] for i in idxs]].to_numpy().reshape(-1),
    }
    if fam_x2 is not None and x2_label is not None:
        data[x2_label] = df[[fam_x2[i] for i in idxs]].to_numpy().reshape(-1)
    return pd.DataFrame(data)


def ols_and_corr(x, y, x_name, y_name, tag):
    """
    Run OLS (y ~ x) + Pearson r.
    Save summary to OUT_DIR/{tag}_summary.txt and plot to OUT_DIR/{tag}_scatter_fit.png
    """
    valid = pd.notna(x) & pd.notna(y)
    x = pd.Series(x[valid], name=x_name)
    y = pd.Series(y[valid], name=y_name)

    Xc = sm.add_constant(x)
    model = sm.OLS(y, Xc).fit()

    r, p = pearsonr(x, y)

    # Save text summary
    txt_path = OUT_DIR / f"schell_cpitch_{tag}_summary.txt"
    with open(txt_path, 'w') as f:
        f.write(model.summary().as_text())
        f.write('\n')
        f.write(f"Pearson r = {r:.4f}, p = {p:.3g}\n")
    print(f"[saved] {txt_path}")

    # Scatter + regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.45, s=16)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model.predict(sm.add_constant(x_line))
    plt.plot(x_line, y_line, linewidth=2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{y_name} vs {x_name}')
    plt.grid(True)
    plt.tight_layout()
    png_path = OUT_DIR / f"schell_cpitch_{tag}_scatter_fit.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[saved] {png_path}")


def multiple_regression(y, x1, x2, y_name, x1_name, x2_name, tag):
    """
    OLS: y ~ x1 + x2.
    Also saves a standardized-betas version.
    Output -> OUT_DIR/{tag}_multiple_regression.txt
    """
    df = pd.DataFrame({y_name: y, x1_name: x1, x2_name: x2}).dropna()

    X = sm.add_constant(df[[x1_name, x2_name]])
    model = sm.OLS(df[y_name], X).fit()

    # Standardized (beta) regression for comparability
    Z = (df - df.mean()) / df.std(ddof=0)
    Xz = sm.add_constant(Z[[x1_name, x2_name]])
    model_z = sm.OLS(Z[y_name], Xz).fit()

    txt_path = OUT_DIR / f"schell_cpitch_{tag}_multiple_regression.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Raw-scale regression ===\n")
        f.write(model.summary().as_text())
        f.write("\n\n=== Standardized (beta) regression ===\n")
        f.write(model_z.summary().as_text())
    print(f"[saved] {txt_path}")


def main():
    df = pd.read_csv(CSV_PATH)
    df, id_col = ensure_id_col(df)

    # Families (probe index -> column)
    idyom_ic = find_columns(df, IDYOM_IC_PREFIX)
    transf_ic = find_columns(df, TRANSFORMER_IC_PREFIX)
    human    = find_columns(df, HUMAN_RATING_PREFIX)

    if not idyom_ic or not transf_ic or not human:
        raise ValueError("Missing required column families. "
                         "Expected 'idyom ic k', 'transformer ic k', 'human rating k' for k=1..15.")

    # 1) IDyOM IC ↔ Human rating
    long1 = to_long(df, human, 'Human Rating', idyom_ic, 'IDyOM IC', id_col=id_col)
    ols_and_corr(
        x=long1['IDyOM IC'],
        y=long1['Human Rating'],
        x_name='IDyOM IC',
        y_name='Human Rating',
        tag='human_vs_idyom_ic'
    )

    # 2) Transformer IC ↔ Human rating
    long2 = to_long(df, human, 'Human Rating', transf_ic, 'Transformer IC', id_col=id_col)
    ols_and_corr(
        x=long2['Transformer IC'],
        y=long2['Human Rating'],
        x_name='Transformer IC',
        y_name='Human Rating',
        tag='human_vs_transformer_ic'
    )

    # 3) Transformer IC ↔ IDyOM IC
    pair = to_long(df, idyom_ic, 'IDyOM IC', transf_ic, 'Transformer IC', id_col=id_col)
    ols_and_corr(
        x=pair['Transformer IC'],
        y=pair['IDyOM IC'],
        x_name='Transformer IC',
        y_name='IDyOM IC',
        tag='transformer_ic_vs_idyom_ic'
    )

    # 4) Two-predictor OLS: Human ~ IDyOM IC + Transformer IC
    long3 = to_long(
        df, human, 'Human Rating',
        idyom_ic, 'IDyOM IC',
        fam_x2=transf_ic, x2_label='Transformer IC',
        id_col=id_col
    )
    multiple_regression(
        y=long3['Human Rating'],
        x1=long3['IDyOM IC'],
        x2=long3['Transformer IC'],
        y_name='Human Rating',
        x1_name='IDyOM IC',
        x2_name='Transformer IC',
        tag='human_on_both_ics'
    )

    print("\nAll analyses complete. Results saved in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
