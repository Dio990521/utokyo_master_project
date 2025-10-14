import pandas as pd
import matplotlib.pyplot as plt
import os

VERSION = "20251015_new_test_5"
PLOT_VALIDATION_DATA = False
COLUMN_TO_PLOT = "similarity"  # similarity, used_budgets, block_similarity, block_reward, step_rewards
TRAIN_WINDOW_SIZE = 100

def plot_training_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "training_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if COLUMN_TO_PLOT not in df.columns:
        print(f"Error: Column '{COLUMN_TO_PLOT}' not found in training data.")
        return

    plt.figure(figsize=(15, 7))

    data_to_plot = df[COLUMN_TO_PLOT].rolling(window=TRAIN_WINDOW_SIZE).mean()

    plt.plot(df.index, data_to_plot, label=f'Training (MA {TRAIN_WINDOW_SIZE} episodes)')
    plt.title(f"Training Performance: {COLUMN_TO_PLOT.replace('_', ' ').title()} (Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel(COLUMN_TO_PLOT.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_validation_data():
    DATA_PATH = os.path.join(f"../training_outputs/{VERSION}/", "validation_data.csv")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Validation data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if COLUMN_TO_PLOT not in df.columns:
        print(f"Error: Column '{COLUMN_TO_PLOT}' not found in validation data.")
        return

    plt.figure(figsize=(15, 7))

    val_agg = df.groupby('step')[COLUMN_TO_PLOT].mean()

    plt.plot(val_agg.index, val_agg.values, 'o-', label='Validation Score', markersize=6)
    plt.scatter(val_agg.index, val_agg.values)

    plt.title(f"Validation Performance: {COLUMN_TO_PLOT.replace('_', ' ').title()}")
    plt.xlabel("Training Timestep")
    plt.ylabel(COLUMN_TO_PLOT.replace('_', ' ').title())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if PLOT_VALIDATION_DATA:
        print(f"Plotting VALIDATION data for version: {VERSION}")
        plot_validation_data()
    else:
        print(f"Plotting TRAINING data for version: {VERSION}")
        plot_training_data()