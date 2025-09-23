# train/plot.py
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import os

VERSION = "_20251001_5"

PLOT_MODE = "episode" #delta_histogram, episode
USE_MOVING_AVERAGE = True
WINDOW_SIZE = 100
PLOT_STYLE = 'line'# Options: 'line' or 'scatter'
COLUMN_TO_PLOT = "similarity"  # similarity, used_budgets, block_similarity, block_reward, step_rewards

plt.figure(figsize=(12, 6))

if PLOT_MODE == 'episode':
    DATA_PATH = f"../training_outputs/{VERSION}/episode_data.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Error: Cannot find the data file at {DATA_PATH}")
        exit()

    df = pd.read_csv(DATA_PATH, index_col="episode")

    if COLUMN_TO_PLOT not in df.columns:
        print(f"Error: Column '{COLUMN_TO_PLOT}' not found.")
        exit()

    if USE_MOVING_AVERAGE:
        data_to_plot = df[COLUMN_TO_PLOT].rolling(window=WINDOW_SIZE).mean()
        print(round(numpy.nanmax(data_to_plot.values), 4))
        #plot_label = f'{window_size}-Episode Moving Average'
        title_suffix = "(Moving Average)"
        plt.plot(data_to_plot.index, data_to_plot)
    else:
        data_to_plot = df[COLUMN_TO_PLOT]
        if PLOT_STYLE == 'scatter':
            #plot_label = 'Raw Data (Scatter)'
            title_suffix = "(Raw Data Scatter)"
            plt.scatter(data_to_plot.index, data_to_plot, alpha=0.4, s=15)
        else:
            #plot_label = 'Raw Data (Line)'
            title_suffix = "(Raw Data Line)"
            plt.plot(data_to_plot.index, data_to_plot)

    plt.xlabel("Episode")
    plt.ylabel(COLUMN_TO_PLOT.replace('_', ' ').title())
    plt.title(f"{COLUMN_TO_PLOT.replace('_', ' ').title()} {title_suffix} during Training")
    plt.grid(True)
    plt.legend()

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
    plt.title("Distribution of Delta Similarity")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()