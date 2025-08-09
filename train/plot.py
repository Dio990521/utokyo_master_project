# train/plot.py

import pandas as pd
import matplotlib.pyplot as plt
import os

VERSION = "_1"

PLOT_MODE = "episode"

if PLOT_MODE == 'episode':
    COLUMN_TO_PLOT = "similarity" # similarity, used_budgets, block_similarity
    DATA_PATH = f"../training_outputs/{VERSION}/episode_data.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Error: Cannot find the data file at {DATA_PATH}")
        exit()

    df = pd.read_csv(DATA_PATH, index_col="episode")

    if COLUMN_TO_PLOT not in df.columns:
        print(f"Error: Column '{COLUMN_TO_PLOT}' not found.")
        exit()

    window_size = 100
    df['moving_avg'] = df[COLUMN_TO_PLOT].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['moving_avg'], label=f'{window_size}-Episode Moving Average')
    plt.xlabel("Episode")
    plt.ylabel(COLUMN_TO_PLOT.title())
    plt.title(f"{COLUMN_TO_PLOT.title()} during Training")
    plt.legend()
    plt.grid(True)

elif PLOT_MODE == 'delta_histogram':
    DATA_PATH = f"../training_outputs/{VERSION}/delta_similarity_data.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Error: Cannot find the data file at {DATA_PATH}")
        exit()

    df_delta = pd.read_csv(DATA_PATH)

    plt.figure(figsize=(12, 7))
    filtered_data = df_delta[df_delta['delta_similarity'].abs() > 1e-6]['delta_similarity']

    plt.hist(filtered_data, bins=100, alpha=0.75, label='Delta Similarity per Step')

    plt.xlabel("Delta Similarity (similarity_t - similarity_t-1)")
    plt.ylabel("Frequency (Count)")
    plt.title("Distribution of Step-level Reward Signals")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()