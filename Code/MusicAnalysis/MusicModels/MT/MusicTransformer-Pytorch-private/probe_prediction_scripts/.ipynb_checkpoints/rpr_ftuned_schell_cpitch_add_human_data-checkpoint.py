import pandas as pd

# Load merged model table
df = pd.read_csv('probe_prediction_results/schell_transformer_optimal_viewpoint_idyom_probs_ics_transformer_probs_ics.csv')

# Parse the vertical human data, skipping lines starting with #
ratings_flat = []
with open('schell96.data') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("rating"):
            continue
        try:
            ratings_flat.append(float(line))
        except ValueError:
            pass

# Number of probes per fragment
ratings_per_fragment = 15

# Chunk into fragments
num_fragments = len(ratings_flat) // ratings_per_fragment
ratings_array = [
    ratings_flat[i*ratings_per_fragment:(i+1)*ratings_per_fragment]
    for i in range(num_fragments)
]

# Reverse each fragment’s ratings (high→low -> low→high)
ratings_array = [chunk[::-1] for chunk in ratings_array]

# sanity checks
assert len(df) == len(ratings_array), f"Row mismatch: df={len(df)} vs human={len(ratings_array)}"
assert len(ratings_flat) % ratings_per_fragment == 0, "Ratings count not a multiple of probes per fragment."

# Build as DataFrame
human = pd.DataFrame(
    ratings_array,
    columns=[f'human rating {i+1}' for i in range(ratings_per_fragment)]
)

# Concatenate to the main DataFrame
df_out = pd.concat([df, human], axis=1)

out_df_path = 'probe_prediction_results/schell_optimal_viewpoint_merged_idyom_transformer_human.csv'
df_out.to_csv(out_df_path, index=False)
print(f"Done! Table saved as {out_df_path}")

