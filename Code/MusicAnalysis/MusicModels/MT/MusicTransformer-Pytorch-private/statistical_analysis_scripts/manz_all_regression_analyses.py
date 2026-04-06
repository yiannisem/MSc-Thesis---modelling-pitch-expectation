import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

CSV_PATH = 'probe_prediction_results/manz_merged_idyom_transformer_human.csv'
OUT_DIR = Path('statistical_analysis_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ols_and_corr(x, y, x_name, y_name, tag):
    """Run OLS (y ~ x) + Pearson r, save summary and a scatter+fit plot."""
    # Drop NAs
    valid = pd.notna(x) & pd.notna(y)
    x = pd.Series(x[valid], name=x_name)
    y = pd.Series(y[valid], name=y_name)

    if len(x) == 0:
        print(f"Skipping {tag}—no valid data.")
        return

    # OLS
    X_const = sm.add_constant(x)
    model = sm.OLS(y, X_const).fit()

    # Pearson correlation
    r, p = pearsonr(x, y)

    # Save text summary
    txt_path = OUT_DIR / f"manz_{tag}_summary.txt"
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
    plt.plot(x_line, y_line, color='red', linewidth=2)
    
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{y_name} vs {x_name}')
    plt.grid(True)
    plt.tight_layout()
    
    png_path = OUT_DIR / f"manz_{tag}_scatter_fit.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[saved] {png_path}")


def multiple_regression(avg, idyom_ic, transf_ic, tag):
    """Run OLS: human_average ~ idyom_ic + transformer_ic (with const)."""
    df = pd.DataFrame({'human_average': avg, 'idyom_ic': idyom_ic, 'transformer_ic': transf_ic}).dropna()
    if len(df) == 0:
        print(f"Skipping {tag}—no valid data.")
        return
        
    X = sm.add_constant(df[['idyom_ic', 'transformer_ic']])
    y = df['human_average']
    model = sm.OLS(y, X).fit()

    # Standardize for beta coefficients
    z = df.copy()
    for col in ['human_average', 'idyom_ic', 'transformer_ic']:
        z[col] = (z[col] - z[col].mean()) / z[col].std(ddof=0)
    
    Xz = sm.add_constant(z[['idyom_ic', 'transformer_ic']])
    yz = z['human_average']
    model_z = sm.OLS(yz, Xz).fit()

    txt_path = OUT_DIR / f"manz_{tag}_multiple_regression.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Raw-scale regression ===\n")
        f.write(model.summary().as_text())
        f.write("\n\n=== Standardized (beta) regression ===\n")
        f.write(model_z.summary().as_text())
    print(f"[saved] {txt_path}")


def main():
    df = pd.read_csv(CSV_PATH)
    
    # Skip the first 5 notes of each chorale to account for initialization bias in the models
    df = df[df['note_index'] >= 7].copy()
    print(f"Loaded {len(df)} rows from {CSV_PATH} after dropping the first 7 notes.")

    # --- Pairwise analyses ---
    ols_and_corr(
        x=df['idyom_ic'],
        y=df['human_average'],
        x_name='IDyOM Information Content',
        y_name='Human Expectedness Rating',
        tag='human_rating_vs_idyom_ic'
    )

    ols_and_corr(
        x=df['transformer_ic'],
        y=df['human_average'],
        x_name='Transformer Information Content',
        y_name='Human Expectedness Rating',
        tag='human_rating_vs_transformer_ic'
    )

    ols_and_corr(
        x=df['idyom_ic'],
        y=df['transformer_ic'],
        x_name='IDyOM Information Content',
        y_name='Transformer Information Content',
        tag='transformer_ic_vs_idyom_ic'
    )

    # --- Multiple regression ---
    multiple_regression(
        avg=df['human_average'],
        idyom_ic=df['idyom_ic'],
        transf_ic=df['transformer_ic'],
        tag='human_rating_on_both_ics'
    )

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
